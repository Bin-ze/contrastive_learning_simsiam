# @Author  : guozebin (guozebin@fuzhi.ai)
# @Desc    :
import lightly
import logging
import torch
import gc
import json
import time
import os
import math
import torch
import torchvision

import torch.nn as nn
import numpy as np

from utils.utils import create_not_exist_path, write_txt_file, set_seed
from utils.const import KEEP_DIGITS_NUM
from interface.gp_output import gen_gp_train_output
from lightly.models.modules.heads import SimSiamPredictionHead
from lightly.models.modules.heads import SimSiamProjectionHead

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

class SimSiam(nn.Module):
    def __init__(
        self, backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(
            num_ftrs, proj_hidden_dim, out_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim
        )

    def forward(self, x):
        # get representations
        f = self.backbone(x).flatten(start_dim=1)
        # get projections
        z = self.projection_head(f)
        # get predictions
        p = self.prediction_head(z)
        # stop gradient
        z = z.detach()
        return z, p

class Trainer(object):
    def __init__(self, dataset_path='', max_epoch=10, device='cpu',
                 batch_size=2, work_dir='', trial_name='', uuid_value=''):
        # const
        self.work_dir = work_dir.replace('\\', '/') if work_dir else './'
        self.model_save_path = f'{work_dir}/runs/models/{uuid_value}'
        self.metric_output_path = f'{work_dir}/runs/metric/trial.txt'
        create_not_exist_path(self.model_save_path)
        self.dataset_path = dataset_path if not dataset_path.endswith('/') else dataset_path[:-1]

        self.trial_name = trial_name
        self.device = torch.device(device)
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.time_use = {}
        self.valid_metric_info = {}
        set_seed(42)

        #config
        self.num_workers = 0
        self.seed = 1
        self.input_size = 256

        # dimension of the embeddings
        self.num_ftrs = 2048
        # dimension of the output of the prediction and projection heads
        self.out_dim = self.proj_hidden_dim = 512
        # the prediction head uses a bottleneck architecture
        self.pred_hidden_dim = 128
        # seed torch and numpy
        torch.manual_seed(0)
        np.random.seed(0)
        # model
        resnet = torchvision.models.resnet50()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.model = SimSiam(backbone, self.num_ftrs, self.proj_hidden_dim, self.pred_hidden_dim, self.out_dim)
        self.model_name = 'SimSiam'

        #loss
        self.criterion = lightly.loss.NegativeCosineSimilarity()
        # scale the learning rate
        lr = 0.05 * batch_size / 256
        # use SGD with momentum and weight decay
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        # to device
        self.model.to(self.device)

        #data pre
        self.dataloader_train_simsiam, self.dataloader_train_refine, self.dataloader_test, self.dataset_test = self.data_prepare()
        num_classes = len(self.dataloader_train_refine.dataset.dataset.classes)

        # val set
        self.valid_per_epoch = max_epoch
        self.refine_epochs = 20
        self.metric_type = 'acc'
        self.net = Classifier(backbone, num_class=num_classes)
        self.net.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.net.fc.bias.data.zero_()


        # train
        self.avg_loss = 0.
        self.avg_output_std = 0.
        self.best_ckpt = [0, {}]

    def data_prepare(self):

        train_dataset = self.dataset_path+'/train/images'
        valid_dataset = self.dataset_path+'/valid/images'
        # define the augmentations for self-supervised learning
        collate_fn = lightly.data.ImageCollateFunction(
            input_size=self.input_size,
            # require invariance to flips and rotations
            hf_prob=0.5,
            vf_prob=0.5,
            rr_prob=0.5,
            # satellite images are all taken from the same height
            # so we use only slight random cropping
            min_scale=0.5,
            # use a weak color jitter for invariance w.r.t small color changes
            cj_prob=0.2,
            cj_bright=0.1,
            cj_contrast=0.1,
            cj_hue=0.1,
            cj_sat=0.1,
        )

        # create a lightly dataset for training, since the augmentations are handled
        # by the collate function, there is no need to apply additional ones here
        dataset_train_simsiam = lightly.data.LightlyDataset(
            input_dir=train_dataset
        )
        # create a dataloader for training
        dataloader_train_simsiam = torch.utils.data.DataLoader(
            dataset_train_simsiam,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=self.num_workers
        )


        #refine dataset
        dataset_train_refine = lightly.data.LightlyDataset(
            input_dir=valid_dataset
        )
        # create a dataloader for training
        dataloader_train_refine = torch.utils.data.DataLoader(
            dataset_train_refine,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=self.num_workers
        )

        # create a torchvision transformation for embedding the dataset after training
        # here, we resize the images to match the input size during training and apply
        # a normalization of the color channel based on statistics from imagenet
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
        cla_dict = dataset_test.dataset.class_to_idx
        self.cla_dict = dict((key, value) for value, key in cla_dict.items())
        json_str = json.dumps(cla_dict, indent=1)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        # create a dataloader for valid
        dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers
        )
        return dataloader_train_simsiam, dataloader_train_refine, dataloader_test, dataset_test

    def train(self):
        train_time = 0
        valid_time = 0

        # loop
        total_time_tic = time.time()
        best_loss = 1
        print('Start Contrastive Learning train')
        for epoch in range(self.max_epoch):
            # train
            train_time_tic = time.time()

            avg_loss, collapse_level= self.train_epoch()
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.model_save_path+'/SimSiam.pth')
            print(f'[Epoch {epoch:3d}] '
                  f'Contrastive Learning Loss = {avg_loss:.2f} | '
                  f'Collapse Level: {collapse_level:.2f} / 1.00')

            train_time += (time.time() - train_time_tic)

            # valid
            valid_time_tic = time.time()
            if epoch == (self.valid_per_epoch - 1):
                print('Refine Contrastive Learning weight to Classifier task')
                self.valid_epoch()
                valid_time += (time.time() - valid_time_tic)

        self.time_use = {
            'total': round(time.time() - total_time_tic, KEEP_DIGITS_NUM),
            'train_time': round(train_time, KEEP_DIGITS_NUM),
            'valid_time': round(valid_time, KEEP_DIGITS_NUM),
        }

        # output metric
        self.output_metric()

        logging.info('train success ...')

    def output_metric(self):
        metric_info = gen_gp_train_output(
            model_name=self.model_name,
            trial_name=self.trial_name,
            metric=self.valid_metric_info,
            model_path=self.model_save_path,
            metric_type=self.metric_type,
            time_use=self.time_use
        )

        # write metric to path
        write_txt_file(metric_info, txt_save_path=self.metric_output_path)
        logging.info(f'save trial.txt succeed -> {self.metric_output_path}')
        logging.info(metric_info)

    def train_epoch(self):
        for (x0, x1), _, _ in self.dataloader_train_simsiam:
            # move images to the gpu
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)

            # run the model on both transforms of the images
            # we get projections (z0 and z1) and
            # predictions (p0 and p1) as output
            z0, p0 = self.model(x0)
            z1, p1 = self.model(x1)

            # apply the symmetric negative cosine similarity
            # and run backpropagation
            loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # calculate the per-dimension standard deviation of the outputs
            # we can use this later to check whether the embeddings are collapsing
            output = p0.detach()
            output = torch.nn.functional.normalize(output, dim=1)

            output_std = torch.std(output, 0)
            output_std = output_std.mean()

            # use moving averages to track the loss and standard deviation
            w = 0.9
            self.avg_loss = w * self.avg_loss + (1 - w) * loss.item()
            self.avg_output_std = w * self.avg_output_std + (1 - w) * output_std.item()

        # the level of collapse is large if the standard deviation of the l2
        # normalized output is much smaller than 1 / sqrt(dim)
        collapse_level = max(0., 1 - math.sqrt(self.out_dim) * self.avg_output_std)

        return self.avg_loss, collapse_level

    def valid_epoch(self):
        pretrained = self.model_save_path+'/SimSiam.pth'
        checkpoint = torch.load(pretrained, map_location="cpu")
        msg = self.net.load_state_dict(checkpoint, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print("=> loaded Contrastive Learning pre-trained model '{}'".format(pretrained))
        self.net.to(self.device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        train_steps = len(self.dataloader_train_refine)
        val_num = len(self.dataset_test)
        for epoch in range(self.refine_epochs):
            # train
            self.net.train()
            running_loss = 0.0
            for step, data in enumerate(self.dataloader_train_refine):
                (images, _), labels, _ = data
                optimizer.zero_grad()
                logits = self.net(images.to(self.device))
                loss = loss_function(logits, labels.to(self.device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            # validate
            self.net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                for val_data in self.dataloader_test:
                    val_images, val_labels, _ = val_data
                    outputs = self.net(val_images.to(self.device))
                    # AUC.compute_metric(outputs,val_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(self.device)).sum().item()

            val_accurate = acc / val_num
            print('[epoch %d] refine_train_loss: %.3f  refine_val_accuracy: %.4f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            # record best ckpt
            psnr = val_accurate
            if psnr > self.best_ckpt[0]:
                self.best_ckpt = [psnr, self.net.state_dict()]
                self.save_model()
        if not self.best_ckpt[-1]:
            self.best_ckpt = [psnr, self.net.state_dict()]

        self.valid_metric_info = {'acc': round(psnr, KEEP_DIGITS_NUM)}

    def save_model(self):
        create_not_exist_path(f'{self.model_save_path}/')
        file_name = f'{self.model_name}_{self.trial_name}.pth'
        ckpt = dict(
            best_ckpt=self.best_ckpt[-1],
            metric_value=self.best_ckpt[0],
            metric_type=self.metric_type,
            classes_dict=self.cla_dict
        )
        torch.save(ckpt, f'{self.model_save_path}/{file_name}', _use_new_zipfile_serialization=False)
        logging.info(f'success save model, {file_name}')


if __name__ == '__main__':
    trainer = Trainer(dataset_path='')
    trainer.train()
