import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline
from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler
from diffusers.models import AutoencoderKL
from diffusers.utils import load_image
import matplotlib.pyplot as plt

pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to("cuda")

pipe.load_lora_weights("jbilcke-hf/sdxl-panorama", weight_name="lora.safetensors")

text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

embedding_path = hf_hub_download(repo_id="jbilcke-hf/sdxl-panorama", filename="embeddings.pti", repo_type="model")
embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
embhandler.load_embeddings(embedding_path)

def create_pano(prompt, dims = (1024,512)):
    prompt="hdri view, {}, in the style of <s0><s1>".format(prompt)
    images = pipe(
        prompt,
        negative_prompt = "overexposed, blur, blurry, blurred",
        width = dims[0], 
        height = dims[1],
        cross_attention_kwargs={"scale": 0.8},
        num_inference_steps = 50,
    ).images
    #your output image
    return images[0]