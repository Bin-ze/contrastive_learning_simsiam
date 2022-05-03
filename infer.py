import torch
import gc
import os
import logging
import torchvision
import lightly
import cv2

from utils.utils import create_not_exist_path, set_seed
from PIL import Image

import torch.nn as nn

class Classifier(nn.Module):
    def __init__(
        self, backbone, num_class,
    ):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat
input_size = 256
infer_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])
model_load_path = '/home/liuguangcan/internship/Contrastive_Learning/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc/SimSiam_autotables-2503-train-1.pth'
ckpt = torch.load(model_load_path, map_location='cpu')
dict_cls = ckpt['classes_dict']
num_classes = len(dict_cls)
resnet = torchvision.models.resnet50()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = Classifier(backbone, num_class=num_classes)
msg = model.load_state_dict(ckpt['best_ckpt'], strict=False)
print("=> loaded Contrastive Learning pre-trained model '{}'".format(model_load_path))
model.to('cuda')
def infer(img,cla,acc):
    model.eval()
    with torch.no_grad():
        pred = torch.squeeze(model(img.to('cuda'))).cpu()
        predict = torch.softmax(pred, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        result = [str(dict_cls[int(predict_cla)]), str("{:.4}".format(predict[predict_cla].numpy()))]
        if result[0]==cla:
            acc+=1
        return result,acc
model.to('cuda')
path='/data/guozebin_data/flower_data/val'
acc=0
m=0
for i in sorted(os.listdir(path)):
    print(i)
    for j in sorted(os.listdir(os.path.join(path,i))):
        m+=1
        img = os.path.join(path,i,j)
        img = cv2.imread(img)
        img = img[:, :, ::-1]
        img = Image.fromarray(img)
        img = infer_transforms(img)
        img = img.unsqueeze(dim=0)
        result,acc = infer(img,i,acc)
print(acc/m)
