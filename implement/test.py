import torch
import gc
import time
import logging
import os
import torchvision
import lightly

import torch.nn as nn

from utils.utils import write_txt_file, write_json_file, set_seed
from utils.const import KEEP_DIGITS_NUM
from interface.gp_output import gen_gp_test_output

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

class Tester(object):
    def __init__(self, dataset_path='', model_path='', used_model='',
                 device='cpu', work_dir=''):
        # const
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.used_model = used_model.replace('.pth', '').replace('.pt', '') + '.pth'
        self.model_load_path = f'{self.model_path}/{self.used_model}'
        self.work_dir = work_dir.replace('\\', '/') if work_dir else './'
        self.predict_output_dir = self.model_path
        self.predict_file_name = f'evaluation_{int(time.time())}.json'
        self.metric_output_path = f'{work_dir}/runs/metric/trial.txt'
        self.device = torch.device(device)
        self.time_use = {}
        self.test_metric_info = {}
        set_seed(42)

        # model
        self.model_name = 'SimSiam'
        # data
        self.input_size = 256
        self.num_workers = 0
        self.batch_size = 32
        self.dataloader_test, self.dataset_test = self.data_prepare()

        # load model
        ckpt = torch.load(self.model_load_path, map_location='cpu')
        dict_cls = ckpt['classes_dict']
        num_classes=len(dict_cls)
        resnet = torchvision.models.resnet50()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.model = Classifier(backbone,num_class=num_classes)
        msg = self.model.load_state_dict(ckpt['best_ckpt'], strict=False)
        print("=> loaded Contrastive Learning pre-trained model '{}'".format(self.model_load_path))

        # to device
        self.model.to(self.device)

    def data_prepare(self):

        valid_dataset = self.dataset_path+'/test/images'
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.input_size, self.input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=lightly.data.collate.imagenet_normalize['mean'],
                std=lightly.data.collate.imagenet_normalize['std'],
            )
        ])

        # create a lightly dataset for embedding
        dataset_test = lightly.data.LightlyDataset(
            input_dir=valid_dataset,
            transform=test_transforms
        )

        # create a dataloader for embedding
        dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers
        )
        return dataloader_test, dataset_test
    def test(self):
        test_time_tic = time.time()

        # test
        self.model.eval()
        with torch.no_grad():
            self.test_epoch()

        self.time_use['test'] = round(time.time() - test_time_tic, KEEP_DIGITS_NUM)

        # 输出predict结果
        write_json_file(self.test_metric_info, f'{self.predict_output_dir}/{self.predict_file_name}')

        # 输出metric
        self.output_metric()

        logging.info('test finished ...')

    def output_metric(self):
        metric_info = gen_gp_test_output(
            model_name=self.model_name,
            metric=self.test_metric_info,
            model_path=self.predict_output_dir,
            predict_file_name=self.predict_file_name,
            time_use=self.time_use
        )
        # write metric to path
        write_txt_file(metric_info, txt_save_path=self.metric_output_path)
        logging.info(f'save trial.txt succeed -> {self.metric_output_path}')
        logging.info(metric_info)

    def test_epoch(self):
        acc = 0.0
        val_num = len(self.dataset_test)
        for val_data in self.dataloader_test:
            val_images, val_labels, _ = val_data
            outputs = self.model(val_images.to(self.device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(self.device)).sum().item()
        val_accurate = acc / val_num
        self.test_metric_info = {'acc': round(val_accurate, KEEP_DIGITS_NUM)}
        print('refine_Classifier_test_accuracy: %.4f' % (val_accurate))