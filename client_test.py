#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : 离线验证阶段启动入口

import sys
import logging
import argparse
import os

from interface import Tester


def main(dataset_path, model_path, used_model, device="cpu", work_dir="./runs/"):
    tester = Tester(dataset_path=dataset_path,
                    model_path=model_path,
                    used_model=used_model,
                    device=device,
                    work_dir=work_dir)
    tester.test()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/guozebin_data/flower_classifier/")
    parser.add_argument("--model_path", type=str, default=r"runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc")
    parser.add_argument("--used_model", type=str, default="SimSiam_autotables-2503-train-1")
    parser.add_argument("--device", type=str, default='cuda')

    args = parser.parse_args()
    logging.info("args: {}".format(args))

    main(args.dataset_path, model_path=args.model_path, used_model=args.used_model,
         device=args.device, work_dir="/app/tianji")
