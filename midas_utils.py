import torch
from PIL import Image
import cv2
import numpy as np


def depth(img, model_name):
    # select model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Models
    midas = torch.hub.load("intel-isl/MiDaS", model_name)
    midas.to(device)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform

    # Prep Image
    cv_image = np.array(img)
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    img = Image.fromarray(formatted)
    return img
