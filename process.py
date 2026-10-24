import argparse
import os
import sys
import glob
import shutil
import joblib as jl
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.InteractVLM import InteractVLMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from utils.utils import convert_contacts
from utils.demo_utils import process_smplx_mesh_with_contacts
from datasets.base_contact_dataset import normalize_cam_params
from preprocess_data.constants import HUMAN_VIEW_DICT, SMPL_TO_SMPLX_MAPPING


def parse_args(args):
    parser = argparse.ArgumentParser(description="InteractVLM chat - Human Contact Only")
    parser.add_argument("--version", default="./trained_models/interactvlm-3d-hcontact-wScene-damon-lemon-rich")
    parser.add_argument("--base_dir", required=True, type=str,
                        help="Base directory containing images in BASE/SEQ/... structure")
    parser.add_argument("--tgt_name", required=True, type=str,
                        help="Base directory containing images in BASE/SEQ/... structure")
    parser.add_argument("--obj", required=False, type=str,
                        help="Base directory containing images in BASE/SEQ/... structure")

    parser.add_argument("--prompt", default="Which body parts are in contact with the {object}? Segment these contact areas.",
                        type=str, help="Prompt template for human contact prediction")
    parser.add_argument("--save_vis", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to save visualization images")
    parser.add_argument('--resume', action='store_true', help='Reuse matching per-image results')
    parser.add_argument('--frame-indices', nargs='+', type=int,
                        help='Optional original numeric frame indices for a representative check')
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def get_image_paths(base_dir):
    """Get all image paths from BASE/SEQ/... structure"""
    image_paths = []
    base_path = base_dir# os.path.join(base_dir, tgt_name)
    
    if not os.path.exists(base_path):
        print(f"Error: BASE directory not found at {base_path}")
        return image_paths
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                # Get relative path from BASE for maintaining structure
                rel_path = os.path.relpath(full_path, base_path)
                image_paths.append((full_path, rel_path))
    
    return sorted(image_paths)


@torch.no_grad()
def main(args):
    args = parse_args(args)
    img_folder = os.path.join(args.base_dir, args.tgt_name)
    
    # Get all image paths
    image_paths = get_image_paths(img_folder)
    if args.frame_indices is not None:
        requested = set(args.frame_indices)
        image_paths = [(p,r) for p,r in image_paths if Path(p).stem.isdigit() and int(Path(p).stem) in requested]
        if {int(Path(p).stem) for p,_ in image_paths} != requested:
            raise ValueError('Requested frame indices are missing')
    if not image_paths:
        print("No images found in the specified directory")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Create output directories under the CRISP repo by default.
    repo_root = Path(__file__).resolve().parents[2]
    default_vert_root = repo_root / "results" / "init" / "contacts"
    default_vis_root = repo_root / "results" / "init" / "contact_vis"

    vert_root = Path(os.getenv("CRISP_CONTACT_ROOT", str(default_vert_root)))
    vis_root = Path(os.getenv("CRISP_CONTACT_VIS_ROOT", str(default_vis_root)))

    vert_base = vert_root / args.tgt_name
    vis_base = vis_root / args.tgt_name
    vert_base.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_base.mkdir(parents=True, exist_ok=True)

    # Keep partial results useful after long-running inference is interrupted.
    model_config = Path(args.version)/'config.json'
    signature = dict(model_path=str(Path(args.version).resolve()),
                     model_config_sha256=hashlib.sha256(model_config.read_bytes()).hexdigest(),
                     object_name=args.obj, prompt=args.prompt, precision=args.precision,
                     load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit)
    pending, signatures = [], {}
    for full_path, rel_path in image_paths:
        provenance = dict(signature, source_image_sha256=hashlib.sha256(Path(full_path).read_bytes()).hexdigest())
        signatures[rel_path] = provenance
        result_path = (vert_base/rel_path).with_suffix('.npz')
        metadata_path = result_path.with_suffix('.json')
        if result_path.exists():
            if not args.resume:
                raise FileExistsError(f'{result_path}; use --resume for matching results')
            if not metadata_path.exists() or json.loads(metadata_path.read_text()).get('provenance') != provenance:
                raise ValueError(f'Existing output has different or missing provenance: {result_path}')
            with np.load(result_path, allow_pickle=False) as result:
                contact = result['pred_contact_3d_smplh']
                if contact.size != 6890 or not np.isfinite(contact).all():
                    raise ValueError(f'Invalid saved contact: {result_path}')
            continue
        pending.append((full_path, rel_path))
    image_paths = pending
    print(f'Pending images: {len(image_paths)}', flush=True)
    if not image_paths:
        return

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    smpl_to_smlpx_mapping = jl.load(SMPL_TO_SMPLX_MAPPING)['matrix']
    smpl_to_smlpx_mapping = torch.tensor(smpl_to_smlpx_mapping).float().cuda()

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
    
    kwargs.update({"train_from_LISA": False})
    kwargs.update({"train_from_LLAVA": False})

    model = InteractVLMForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    # Setup for human contact
    #camera_params_humans = HUMAN_VIEW_DICT[model.config.hC_sam_view_type]['cam_params']
    camera_params_humans = HUMAN_VIEW_DICT[model.config.hC_sam_view_type]['cam_params']
    view_names = list(camera_params_humans.keys())
    cam_params = [normalize_cam_params(camera_params_humans[view]) for view in view_names]
    cam_params = torch.stack(cam_params).unsqueeze(0).cuda()
    
    # Fixed SAM paths for human body
    base_path = './data/hcontact_vitruvian/'
    sam_image_paths_template = [f'{base_path}/body_render_norm_{view}.png' for view in view_names]
    overlay_sam_paths_template = [f'{base_path}/smplh_body_render_blue_{view}.png' for view in view_names]
    mask_color = (np.array([1.0, 0.15, 0.10]) * 255).astype(np.uint8)

    # Process each image
    for full_path, rel_path in image_paths:
        print(f"\nProcessing: {rel_path}")
        object_name = args.obj
        
        
        # Format prompt with object name
        prompt = args.prompt.format(object=object_name)
        
        # Prepare conversation
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        prompt_with_token = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt_with_token = prompt_with_token.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt_with_token)
        conv.append_message(conv.roles[1], "")
        prompt_final = conv.get_prompt()
        print(prompt_final)

        # Load and preprocess image
        image_llava_np = cv2.imread(full_path)
        if image_llava_np is None:
            print(f"Error: Could not load image {full_path}")
            continue
            
        image_llava_np = cv2.cvtColor(image_llava_np, cv2.COLOR_BGR2RGB)

        image_clip = (
            clip_image_processor.preprocess(image_llava_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )

        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        # Prepare SAM multiview images
        sam_image = [Image.open(sam_img) for sam_img in sam_image_paths_template]
        sam_multiview = np.stack([np.asarray(sam_img) for sam_img in sam_image], axis=0)
        valid_masks_region = [(sam_img.sum(axis=-1) < 255*3).astype(np.uint8) for sam_img in sam_multiview]
        sam_multiview = [transform.apply_image(sam_img) for sam_img in sam_multiview]
        resize_list = [sam_multiview[0].shape[:2]]
        sam_multiview = torch.stack([preprocess(torch.from_numpy(sam_img).permute(2, 0, 1).contiguous()) for sam_img in sam_multiview])
        sam_multiview = sam_multiview.unsqueeze(0).cuda()

        if args.precision == "bf16":
            sam_multiview = sam_multiview.bfloat16()
            cam_params_curr = cam_params.bfloat16()
        elif args.precision == "fp16":
            sam_multiview = sam_multiview.half()
            cam_params_curr = cam_params
        else:
            sam_multiview = sam_multiview.float()
            cam_params_curr = cam_params

        input_ids = tokenizer_image_token(prompt_final, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # Run model
        output = model.evaluate(
            image_clip,
            sam_multiview,
            input_ids,
            cam_params_curr,
            resize_list,
            original_size_list=resize_list,
            lift2d_dict_path=None,
            contact_type='hcontact-wScene',
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        
        output_ids, pred_masks = output["output_ids"], output["pred_masks"]
        pred_contact_3d = output["pred_contact_3d"].detach()
        pred_contact_3d_raw = output["pred_contact_3d_raw"].detach()
        
        # Decode text output
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print(f'Response: {text_output}')
        print(f'Non-zero contact vertices: {pred_contact_3d[pred_contact_3d != 0].shape[0]}')
        
        # Save vertices
        # Create directory structure in VERT matching the source structure
        vert_dir = vert_base / os.path.dirname(rel_path)
        vert_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with same name as original image but with .npz extension
        base_name = os.path.splitext(os.path.basename(full_path))[0]
        vert_path = vert_dir / f"{base_name}.npz"
        
        # Convert to SMPLX format
        pred_contact_3d_smplx = convert_contacts(pred_contact_3d, smpl_to_smlpx_mapping)
        pred_contact_3d_raw_smplx = convert_contacts(pred_contact_3d_raw, smpl_to_smlpx_mapping)
        temporary_path = vert_path.with_suffix('.tmp.npz')
        np.savez(str(temporary_path),
                pred_contact_3d_smplh=pred_contact_3d.cpu(),
                pred_contact_3d_raw_smplh=pred_contact_3d_raw.cpu(),
                pred_contact_3d_smplx=pred_contact_3d_smplx.cpu(),
                pred_contact_3d_raw_smplx=pred_contact_3d_raw_smplx.cpu()
                )
        vert_path.with_suffix('.json').write_text(json.dumps(dict(
            provenance=signatures[rel_path],response=text_output),indent=2))
        temporary_path.replace(vert_path)
        print(f"Saved vertices to: {vert_path}")
        
        # Save visualization if requested
        if args.save_vis and pred_masks is not None:
            pred_masks = pred_masks[0]
            if pred_masks.shape[0] > 0:
                # Create directory structure in VIS matching the source structure
                vis_dir = vis_base / os.path.dirname(rel_path)
                vis_dir.mkdir(parents=True, exist_ok=True)
                
                overlay_images = []
                for i, pred_mask in enumerate(pred_masks):
                    pred_mask = pred_mask.detach().cpu().numpy()
                    pred_mask = pred_mask > 0.3
                    
                    overlay_sam_view_path = overlay_sam_paths_template[i]
                    overlay_sam = cv2.imread(overlay_sam_view_path)
                    overlay_sam = cv2.cvtColor(overlay_sam, cv2.COLOR_BGR2RGB)

                    valid_mask_region = valid_masks_region[i]
                    pred_mask = np.logical_and(pred_mask, valid_mask_region)
                    
                    # Expand pred_mask to match the RGB channels
                    pred_mask_3d = np.stack([pred_mask] * 3, axis=2)
                    
                    # Apply the mask
                    overlay_sam = np.where(pred_mask_3d, 
                                        overlay_sam * 0.5 + mask_color * 0.5,
                                        overlay_sam)
                    overlay_sam = np.clip(overlay_sam, 0, 255).astype(np.uint8)
                    overlay_sam = cv2.cvtColor(overlay_sam, cv2.COLOR_RGB2BGR)
                    overlay_images.append(overlay_sam)
                
                # Create 2x2 grid
                if len(overlay_images) >= 4:
                    h, w = overlay_images[0].shape[:2]
                    grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
                    grid[:h, :w] = overlay_images[0]  # top-left
                    grid[:h, w:] = overlay_images[1]  # top-right
                    grid[h:, :w] = overlay_images[2]  # bottom-left
                    grid[h:, w:] = overlay_images[3]  # bottom-right
                    
                    # Save visualization
                    vis_path = vis_dir / f"{base_name}_vis.jpg"
                    cv2.imwrite(str(vis_path), grid)
                    print(f"Saved visualization to: {vis_path}")
                    
                    # Also save SMPLX mesh with contacts if requested
                    smplx_path = vis_dir / f"{base_name}_smplx_with_contacts.obj"
                    process_smplx_mesh_with_contacts(
                        pred_contact_3d_smplx,
                        str(smplx_path),
                        contact_threshold=0.1,
                        gender='neutral'
                    )
                    print(f"Saved SMPLX mesh to: {smplx_path}")

    print("\nProcessing complete!")
    print(f"Vertices saved in: {vert_base}")
    if args.save_vis:
        print(f"Visualizations saved in: {vis_base}")


if __name__ == "__main__":
    main(sys.argv[1:])
