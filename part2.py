import torch
import numpy as np
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image


controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0-small",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda:1")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda:1")
pipe.load_lora_weights("jbilcke-hf/sdxl-panorama", weight_name="lora.safetensors")
pipe.enable_model_cpu_offload()


def create_pano_depth(prompt, non_depth_pano):
    prompt="hdri view, {}, in the style of <s0><s1>".format(prompt)
    
    controlnet_conditioning_scale = 0.8  # recommended for good generalization
    width, height = non_depth_pano.size
    depth_image = Image.open('depth.png').resize((width, height))

    images = pipe(
        prompt,
        image=non_depth_pano,
        control_image=depth_image,
        negative_prompt = "overexposed, blur, blurry, blurred",
        strength=0.71,
        num_inference_steps=80,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images
    return images[0]
