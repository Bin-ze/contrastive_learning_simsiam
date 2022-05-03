import torch
import gc
import os
import logging
import torchvision
import lightly

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

class Infer(object):
    def __init__(self, model_path='', used_model='',
                 device='cpu', work_dir=''):
        # const
        self.work_dir = work_dir
        self.model_path = model_path
        self.used_model = used_model.replace('.pth', '').replace('.pt', '') + '.pth'
        self.model_load_path = f'{self.model_path}/{self.used_model}'
        self.work_dir = work_dir.replace('\\', '/') if work_dir else './'
        self.infer_result_save_path = f'{self.work_dir}/runs/infer'
        self.extract_frame_save_path = f'{work_dir}/runs/dataset/infer_video'
        create_not_exist_path(self.infer_result_save_path)
        create_not_exist_path(self.extract_frame_save_path)
        self.device = torch.device(device)
        self.time_use = {}
        set_seed(42)

        # model
        self.model_name = 'SimSiam'

        # load model
        ckpt = torch.load(self.model_load_path, map_location='cpu')
        self.dict_cls = ckpt['classes_dict']
        num_classes = len(self.dict_cls)
        resnet = torchvision.models.resnet50()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.model = Classifier(backbone, num_class=num_classes)
        msg = self.model.load_state_dict(ckpt['best_ckpt'], strict=False)
        print("=> loaded Contrastive Learning pre-trained model '{}'".format(self.model_load_path))
        # to device
        self.model.to(self.device)
        #data
        self.input_size = 256
        self.infer_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.input_size, self.input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=lightly.data.collate.imagenet_normalize['mean'],
                std=lightly.data.collate.imagenet_normalize['std'],
            )
        ])

    def infer(self, img):
        img = Image.fromarray(img)
        img = self.infer_transforms(img)
        img = img.unsqueeze(dim=0)
        self.model.eval()
        with torch.no_grad():
            pred = torch.squeeze(self.model(img.to('cuda'))).cpu()
            predict = torch.softmax(pred, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            result = [str(self.dict_cls[int(predict_cla)]), str("{:.4}".format(predict[predict_cla].numpy()))]
        logging.info('infer finished ...')

        return result
