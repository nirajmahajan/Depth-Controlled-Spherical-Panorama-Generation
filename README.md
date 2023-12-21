# 360-Panorama Generation with Stable Diffusion

## Introduction

This project demonstrates the generation of text-controlled 360-degree panoramas using Stable Diffusion. I have implemented seam-tiling to achieve coherent panoramas and also experimented with depth conditioning using a ControlNet. This README compares the results of panoramas with and without depth conditioning and seam-tiling.

<figure style="text-align: center;">
    <img src="images/raw/prompt0/Prompt0_non_depth_pano.jpg" alt="alt text" width="512" height="256"/>
    <figcaption>A non coherent panorama with inconsistent seams</figcaption>
</figure>

<figure style="text-align: center;">
    <img src="images/raw/prompt0/Prompt0_cleaned_non_depth_pano.jpg" alt="alt text" width="512" height="256"/>
    <figcaption>A coherent panorama: The right and left ends of a coherent panorama match seamlessly.</figcaption>
</figure>

## Requirements

To set up this project, run the following commands:
```
pip install diffusers transformers accelerate safetensors huggingface_hub
git clone https://github.com/replicate/cog-sdxl cog_sdxl
```

## Generating a Prompt Conditioned Panorama

For generating 360-degree panoramas, I utilized the LoRA weights from [this model](https://huggingface.co/jbilcke-hf/sdxl-panorama). Initially, the seams of these panoramas were not coherent. To address this, I rolled the panorama by half its width, placing the incoherent transition in the centre. Then, by masking and inpainting, the center using XL Stable Diffusion, a coherent panorama was achieved.

<figure style="text-align: center;">
    <img src="images/raw/prompt0/Prompt_non_depth0_image_rolled.jpg" alt="alt text" width="512" height="256"/>
    <figcaption>A rolled image. The centre has an inconsistent transition, which is fixed using image inpainting. </figcaption>
</figure>


## Generating a Prompt and Depth Conditioned Panorama

For depth-conditioned panoramas, I used the [ControlNet pipeline](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl). Similar to the previous method, I corrected the non-coherent seams through rolling and inpainting.

## Results

I have tested this methodology with the following prompts:

1. 'A dining room in Persia'
2. 'Inside Bag End'
3. 'Inside the cantina in Star Wars'
4. 'Inside a pirate ship'
5. 'Inside a prison'

Below are the results for each prompt:

![Dining Room in Persia](images/Prompt_0.jpg)
![Inside Bag End](images/Prompt_1.jpg)
![Inside the Cantina](images/Prompt_2.jpg)
![Inside a Pirate Ship](images/Prompt_3.jpg)
![Inside a Prison](images/Prompt_4.jpg)


## Usage of Code

To run the code, execute the following command:
```
python3 main.py
```


`main.py` contains the prompts, which can be edited as needed.

- `part1.py`: Contains `create_pano` function to generate panoramas using a prompt.
- `part2.py`: Contains `create_pano_depth` function for generating panoramas using prompt and depth information.
- `cleaner.py`: Contains `clean_pano` function for seam-stitching.

The generated images are stored in the `images/` directory. Use images from `images/raw/` to render panoramas at [360 Panorama Web Viewer](https://renderstuff.com/tools/360-panorama-web-viewer/).

