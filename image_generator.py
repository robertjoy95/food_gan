import torch
from torchvision.utils import save_image
from model_classes import Generator
from pathlib import Path
from constants import *

def generate_images(model_name="all"):
    # given the path to the desired model, save images in a new folder
    model_save_path = ""
    if model_name == "all":
        model_save_path = Path("models/generator_model.pth")
    else:
        model_save_path = Path(f"models/{model_name}_generator_model.pth")
    image_save_path = Path(f"data/results/{model_name}")
    image_save_path.mkdir(exist_ok=True)
    generator = Generator(ngpu)
    generator.load_state_dict(torch.load(f=model_save_path))
    # generate fake images
    noise = torch.randn(10, nz, 1, 1, device="cpu")
    output = generator(noise)
    print(len(output))

    for i, img in enumerate(output):
        save_image(img, image_save_path / f"{model_name}_{i}.jpeg")

if __name__ == "__main__":
    generate_images("all")
        
        

