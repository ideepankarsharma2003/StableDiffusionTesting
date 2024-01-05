import torch
import PIL
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", 
    # "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
)
pipe = pipe.to("cuda")


def reconstruct_image(init_image, prompt, guidance, strength)->PIL.Image.Image:
    global pipe;
    image = pipe(prompt, image=init_image, strength=strength, guidance_scale=guidance, num_inference_steps=50).images[0]
    return image # PIL.Image.Image
