from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")


def clean_pano(image, prompt, name = "Run0"):
    image = np.asarray(image)/255.
    height, width,_ = image.shape
    image = np.roll(image, width//2, axis = 1)
    mask = np.zeros((height, width))
    w = width//15
    mask[:,(width//2)-w:(width//2)+w] = 1

    generator = torch.Generator(device="cuda").manual_seed(0)

    plt.imsave(name+'_image_rolled.jpg', image)
    # plt.imsave(name+'_mask.jpg', mask, cmap = 'gray')

    image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=4.0,
        num_inference_steps=80,  # steps between 15 and 30 work well for us
        strength=0.95,  # make sure to use `strength` below 1.0
        generator=generator,
        negative_prompt = "overexposed, blur, blurry, blurred",
    ).images[0]

    image = np.asarray(image.resize((width, height)))/255.
    image = np.roll(image, width//2, axis = 1)

    return Image.fromarray(np.uint8(image*255))

