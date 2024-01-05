import streamlit as st
from PIL import Image
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

@st.cache(allow_output_mutation=True)
def reconstruct_image(init_image, prompt, guidance, strength)->PIL.Image.Image:
    # global pipe;
    image = pipe(prompt, image=init_image, strength=strength, guidance_scale=guidance, num_inference_steps=50).images[0]
    return image # PIL.Image.Image


# from utils.stable_diffusion import reconstruct_image


bg_image= st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
prompt= st.sidebar.text_input(label="prompt", value="reconstruct the given image and fill the empty portions, 16k")
guidance= st.sidebar.number_input("guidance(1-20)", value=7, min_value=1, max_value=20)
strength= st.sidebar.number_input("strength(0.01-1.0)", value=0.3, min_value=0.01, max_value=1.0)
button= st.sidebar.button("Reconstruct")

if bg_image and button:
    init_image= Image.open(bg_image)
    st.subheader("Original Image")
    st.image(init_image)
    image= reconstruct_image(
        init_image=init_image,
        prompt=prompt,
        guidance=guidance,
        strength=strength
    )
    st.subheader("Reconstructed Image")
    st.image(image)
