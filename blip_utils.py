from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
from models.blip_vqa import blip_vqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 384
transform = transforms.Compose(
    [
        transforms.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)

model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"

model = blip_decoder(pretrained=model_url, image_size=384, vit="large")
model.eval()
model = model.to(device)

image_size_vq = 480
transform_vq = transforms.Compose(
    [
        transforms.Resize(
            (image_size_vq, image_size_vq), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)

model_url_vq = r"C:\Users\cobanov-dd\Downloads\model__vqa.pth"

model_vq = blip_vqa(pretrained=model_url_vq, image_size=480, vit="base")
model_vq.eval()
model_vq = model_vq.to(device)


def inference(raw_image, strategy):

    image = transform(raw_image).unsqueeze(0).to(device)
    with torch.no_grad():
        if strategy == "Beam search":
            caption = model.generate(
                image, sample=False, num_beams=3, max_length=20, min_length=5
            )
        else:
            caption = model.generate(
                image, sample=True, top_p=0.9, max_length=20, min_length=5
            )
        return "caption: " + caption[0]

