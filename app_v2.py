import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoPipelineForImage2Image

# from streamlit docs
@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    # Initialize the model outside the app
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    )
    pipe = pipe.to("cuda")
    return pipe

pipe = load_model()


# Function to reconstruct image using the global model
# @st.cache
# def reconstruct_image(init_image, prompt, guidance, strength):
    
#     image = pipe(prompt, image=init_image, strength=strength, guidance_scale=guidance, num_inference_steps=50).images[0]
#     return image

# Streamlit app
def main():
    st.sidebar.header("Image Reconstruction App")
    
    # Upload image
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    
    # Input parameters
    prompt = st.sidebar.text_input(label="Prompt", value="reconstruct the given image and fill the empty portions, 16k")
    guidance = st.sidebar.number_input("Guidance (1-20)", value=7, min_value=1, max_value=20)
    strength = st.sidebar.number_input("Strength (0.01-1.0)", value=0.3, min_value=0.1, max_value=1.0)
    
    # Reconstruction button
    button = st.sidebar.button("Reconstruct")
    
    if bg_image and button:
        init_image = Image.open(bg_image).convert("RGB")
        # init_image = init_image.resize((768, 512))
        
        st.subheader("Original Image")
        st.image(init_image)
        
        # Reconstruct the image using the global model
        # reconstructed_image = reconstruct_image(
        #     init_image=init_image,
        #     prompt=prompt,
        #     guidance=guidance,
        #     strength=strength
        # )
        reconstructed_image = pipe(prompt, image=init_image, strength=strength, guidance_scale=guidance, num_inference_steps=70).images[0]
    
        
        st.subheader("Reconstructed Image")
        st.image(reconstructed_image)

if __name__ == "__main__":
    main()
