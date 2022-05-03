#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : 推理阶段启动入口


import sys
import logging
import argparse

from interface import Infer
from interface.infer_service import InferService


def start_service(model_path, used_model, device, service_port, work_dir='',
                  service_config=None):
    if not service_config:
        service_config = {
            "service_route": "autotable/predict",
            "service_port": service_port,
            "app_name": "{}_{}".format("autotable", service_port)
        }

    infer_interface = Infer(model_path=model_path, work_dir=work_dir,
                            used_model=used_model, device=device)
    infer_service = InferService(infer_interface=infer_interface, service_config=service_config, work_dir=work_dir)
    infer_service.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=r"runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc")
    parser.add_argument("--used_model", type=str, default="SimSiam_autotables-2503-train-1")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--service_port", type=int, default=8080)

    args = parser.parse_args()
    logging.info("args: {}".format(args))

    start_service(model_path=args.model_path,
                  used_model=args.used_model,
                  device=args.device,
                  service_port=args.service_port,
                  work_dir='/app/tianji')
