import os
import csv
import glob
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import v2
from models.deit import deit_base_patch16_224


DeiT = deit_base_patch16_224(pretrained=True)
DeiT.head = nn.Linear(768, 3)
DeiT.load_state_dict(torch.load("checkpoints/PNEUMONIA/DeiT/CrossEntropyLoss/1718026792/best.pth"))

val_transform = v2.Compose([v2.ToImagePIL(),
                            v2.Resize(256, antialias=True),
                            v2.CenterCrop(224),
                            v2.ToTensor(),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])


def inference(model, data_path):
    data = []
    model.eval()
    device = torch.device("cuda:0")
    model.to(device)
    header = ["file_name", "class_id"]
    image_paths = glob.glob(os.path.join(data_path, "*.jpg"))
    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            image = val_transform(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            pred = model(image)
            _, pred_label = torch.max(pred, 1)
            p = path.split("/")[-1]
            data.append([p, pred_label.item()])

    with open("result.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


inference(DeiT, "datasets/Pneumonia/test_images")

