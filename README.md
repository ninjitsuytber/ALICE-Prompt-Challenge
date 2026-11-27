# ALICE-Prompt-Challenge
A locally run Reverse Prompt Mini Game

# Tech Stack
Nothing special actually， just a simple UI and a locally run image generation model, then compare the generated image with a 
##### As for image generation model, I'm actually using a local CUDA GPU model: Lykon/dreamshaper-8 accelerated with ByteDance/Hyper-SD (Hyper-SD15-8steps-CFG ┃ lora.safetensors)

# Image similarity check engine
The image similarity engine will score each generated image from 0% to 100% against the reference image by combining ResNet50 deep features for semantic content (45%) and neural style/texture (20%), CIE-LAB color distribution and grid matching for color palette (20%), and luminance, edge, and SSIM structure comparison for composition (15%).
