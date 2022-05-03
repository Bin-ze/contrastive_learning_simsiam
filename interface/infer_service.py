from typing import Dict, Union
import time
import base64
import os
import logging

from drpc.web import HttpServer

from utils.utils import JsonCommonEncoder
from utils.service_tools import get_host_ip
from utils.image_io import decode_base64_image
from utils.visual import get_visual_result, get_localed_visual_data


class InferService(object):

    def __init__(self, infer_interface, service_config=None, client_max_size=1024*1024*100, work_dir='', **kwargs):
        self.work_dir = work_dir
        if not work_dir:
            self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.infer_interface = infer_interface
        self.client_max_size = client_max_size

        # config
        self.service_config = service_config
        self.service_port = self.service_config["service_port"]
        self.service_route = self.service_config["service_route"]
        self.host_ip = get_host_ip()
        if not service_config:
            self.service_config = {
                "service_route": "autotable/predict",
                "service_port": 8086,
                "app_name": "{}_{}".format("autotable", 8086)
            }
        assert (not self.service_config["service_port"] <= 0 and self.service_config["service_port"] <= 65535)

        logging.info(f'service_ip: {self.host_ip}')
        logging.info(f'service_port: {self.service_port}')
        logging.info(f'service_url: {self.service_route}')

        # encoder
        self.json_encoder = JsonCommonEncoder()

    async def predict(self, content, image_type='image', need_visual_data=False, need_visual_result=False):
        """
        HTTP: /autotable/predict POST
        """
        predicts = []
        if isinstance(content, (list, tuple)):
            for image in content:
                predicts.append(self.infer_single_sample(image, image_type, need_visual_data, need_visual_result))
        else:
            predicts.append(self.infer_single_sample(content, image_type, need_visual_data, need_visual_result))

        return predicts

    def infer_single_sample(self, content, image_type, need_visual_data, need_visual_result):
        image = decode_base64_image(content)
        infer_data = self.infer_interface.infer(image)
        infer_data = self.json_encoder.default(infer_data)

        # 可视化
        if need_visual_data or need_visual_result:
            # image-mask
            visual_data = get_localed_visual_data(infer_data, output_type='text')

            if need_visual_result:
                visual_result = get_visual_result(visual_data, image)
            else:
                visual_result = None
        else:
            visual_data = {}
            visual_result = None

        result = {
            'infer_data': infer_data,
            'visual_data': visual_data,
            'visual_result': visual_result
        }
        return result

    def run(self):
        httpserver = HttpServer()
        httpserver.register(self)
        httpserver.run(port=self.service_port, app_config={'client_max_size': self.client_max_size})
