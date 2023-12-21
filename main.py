import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from part1 import create_pano
from part2 import create_pano_depth
from cleaner import clean_pano

prompts = [
    'A dining room in Persia',
    'inside Bag End',
    'inside the cantina in star wars',
    'inside a pirate ship',
    'Inside a prison',
]

for i, p in enumerate(prompts):
    os.makedirs('images/raw/prompt{}'.format(i), exist_ok=True)
    print("\nRunning for prompt:", p)
    print("Creating Panorama without Depth Information:")
    non_depth_pano = create_pano(p, dims = (1024,512))
    print("Creating Panorama with Depth Information:")
    depth_pano = create_pano_depth(p, non_depth_pano)
    print("Performing seam stitching 1")
    cleaned_non_depth_pano = clean_pano(non_depth_pano, p, name = "images/raw/prompt{}/Prompt_non_depth{}".format(i,i))
    print("Performing seam stitching 2")
    cleaned_depth_pano = clean_pano(depth_pano, p, name = "images/raw/prompt{}/Prompt_depth{}".format(i,i))
    name = "Prompt{}_".format(i)


    non_depth_pano.save('images/raw/prompt{}/'.format(i)+name + 'non_depth_pano.jpg')
    depth_pano.save('images/raw/prompt{}/'.format(i)+name + 'depth_pano.jpg')
    cleaned_depth_pano.save('images/raw/prompt{}/'.format(i)+name + 'cleaned_depth_pano.jpg')
    cleaned_non_depth_pano.save('images/raw/prompt{}/'.format(i)+name + 'cleaned_non_depth_pano.jpg')

    # Create a 2x2 subfigure
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # Display the images in each subplot
    axes[0, 0].imshow(non_depth_pano)
    axes[0, 0].set_title('Base Pano')
    axes[0, 0].set_xticks([]), axes[0, 0].set_yticks([])


    axes[0, 1].imshow(cleaned_non_depth_pano)
    axes[0, 1].set_title('Stitched Base Pano')
    axes[0, 1].set_xticks([]), axes[0, 1].set_yticks([])


    axes[1, 0].imshow(depth_pano)
    axes[1, 0].set_title('Depth Pano')
    axes[1, 0].set_xticks([]), axes[1, 0].set_yticks([])


    axes[1, 1].imshow(cleaned_depth_pano)
    axes[1, 1].set_title('Stitched Depth Pano')
    axes[1, 1].set_xticks([]), axes[1, 1].set_yticks([])


    # Adjust layout and show the plot
    plt.tight_layout()
    plt.suptitle(p)
    plt.savefig('images/Prompt_{}.jpg'.format(i))