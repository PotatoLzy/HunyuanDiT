from pathlib import Path

from loguru import logger

from mllm.dialoggen_demo import DialogGen
from hydit.config import get_args
from hydit.inference_relight import End2End

from PIL import Image
import numpy as np
import torch
from briarmbg import BriaRMBG
import os


rmbg = BriaRMBG.from_pretrained("/home/zhaoyi/media/vllm_ckpt/RMBG-1.4")
device = "cuda" if torch.cuda.is_available() else "cpu"
rmbg = rmbg.to(device=device, dtype=torch.float32)


def inferencer():
    args = get_args()

    args.relight_mode = "fg"

    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    # Load models
    gen = End2End(args, models_root_path)

    # Try to enhance prompt
    if args.enhance:
        logger.info("Loading DialogGen model (for prompt enhancement)...")
        enhancer = DialogGen(str(models_root_path / "dialoggen"), args.load_4bit)
        logger.info("DialogGen model loaded.")
    else:
        enhancer = None

    return args, gen, enhancer

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h

@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


image_path = "/home/zhaoyi/media/dataset/ffhq/1024raw_image/36000/36034.png"

image_path = "/home/zhaoyi/media/dataset/ffhq/1024raw_image/36000/36444.png"
# image_path = "/home/zhaoyi/media/dataset/ffhq/1024raw_image/36000/36165.png"


if __name__ == "__main__":

    args, gen, enhancer = inferencer()

    if enhancer:
        logger.info("Prompt Enhancement...")
        success, enhanced_prompt = enhancer(args.prompt)
        if not success:
            logger.info("Sorry, the prompt is not compliant, refuse to draw.")
            exit()
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
    else:
        enhanced_prompt = None
    # Load image to inference
    if not os.path.exists(image_path):
        raise FileNotFoundError("The path of image: {image_path} does not exist.")
    
    
    # Run inference
    logger.info("Generating images...")
    height, width = args.image_size

    
    input_fg = Image.open(image_path)
    input_fg = np.array(input_fg)
    input_fg = resize_without_crop(input_fg, height, width)

    logger.info("Extract foreground...")
    input_fg, matting = run_rmbg(input_fg)

    # left light
    gradient = np.linspace(255, 0, width)
    image = np.tile(gradient, (height, 1))
    input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

    # right light
    gradient = np.linspace(0, 255, width)
    image = np.tile(gradient, (height, 1))
    input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

    
    input_fg = torch.tensor(input_fg).permute(2,0,1).float() / 255
    input_bg = torch.tensor(input_bg).permute(2,0,1).float() / 255

    input_fg = torch.cat([input_fg] * (input_bg.shape[0] // input_fg.shape[0]), dim=0)
    # new_sample = torch.cat([input_bg, input_fg], dim=1)
    input_fg = input_fg.unsqueeze(0).to(device).half()
    input_bg = input_bg.unsqueeze(0).to(device).half()
    input_bg = None
    
    results = gen.predict(args.prompt,
                          height=height,
                          width=width,
                          image=[input_fg, input_bg],
                          seed=args.seed,
                          enhanced_prompt=enhanced_prompt,
                          negative_prompt=args.negative,
                          infer_steps=args.infer_steps,
                          guidance_scale=args.cfg_scale,
                          batch_size=args.batch_size,
                          src_size_cond=args.size_cond,
                          use_style_cond=args.use_style_cond,
                          )
    images = results['images']

    # Save images
    save_dir = Path('results')
    save_dir.mkdir(exist_ok=True)
    # Find the first available index
    all_files = list(save_dir.glob('*.png'))
    if all_files:
        start = max([int(f.stem) for f in all_files]) + 1
    else:
        start = 0

    for idx, pil_img in enumerate(images):
        save_path = save_dir / f"{idx + start}.png"
        pil_img.save(save_path)
        logger.info(f"Save to {save_path}")
