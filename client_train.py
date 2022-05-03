#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : 训练阶段启动入口
import os
import sys
import json
import logging
import argparse
import uuid

from utils.utils import json_decode
from interface import Trainer

debug = True
uuid_value = '91e2c5a9-35e4-45ee-90e4-c53d85558bbc' if debug else uuid.uuid4()


def main(dataset_path, annotation_data, trial_name='',
         device="cpu", work_dir=""):
    trainer = Trainer(dataset_path=dataset_path,
                      max_epoch=30,
                      device=device,
                      batch_size=32,
                      work_dir=work_dir,
                      trial_name=trial_name,
                      uuid_value=uuid_value)
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/guozebin_data/flower_classifier/")
    parser.add_argument("--device", type=str, default="cuda")
    # trial_name用于生成用于保存模型的文件名 = {model_name}_{trial_name}.pth
    parser.add_argument("--trial_name", type=str, default="autotables-2503-train-1")
    parser.add_argument("--annotation_data", type=str, default="{'advance_settings':{}}")

    args = parser.parse_args()
    logging.info("args: {}".format(args))

    main(dataset_path=args.dataset_path,
         trial_name=args.trial_name,
         annotation_data=json_decode(args.annotation_data),
         device=args.device,
         work_dir="/app/tianji")
