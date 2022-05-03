# @Author  : zhangyouyuan (zyyzhang@fuzhi.ai)
# @Desc    :

import netifaces
import requests
import json as http_json
import json
import numpy as np
import datetime

from functools import partial
from requests.models import complexjson
from utils.utils import JsonCommonEncoder


def default(self, obj):
    if isinstance(obj, datetime.datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(obj, datetime.date):
        return obj.strftime("%Y-%m-%d")
    elif isinstance(obj, datetime.time):
        return obj.strftime("%H:%M:%S")
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return round(float(obj), 4)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tuple, list)):
        new_obj = []
        for i in obj:
            new_obj.append(self.default(i))
        return new_obj
    elif isinstance(obj, bytes):
        return str(obj, encoding='utf-8')
    elif isinstance(obj, (str, int, float)):
        return obj
    else:
        return json.JSONEncoder.default(self, obj)


def get_host_ip():
    """
    获取机器ip
    :return:
    """
    try:
        gws = netifaces.gateways()
        net_name = gws['default'][netifaces.AF_INET][1]
        info = netifaces.ifaddresses(net_name)
        ip = info[netifaces.AF_INET][0]['addr']
    except Exception as exp:
        print(exp)
        ip = None
    return ip


class HttpClient(object):

    def __init__(self, url: str):
        complexjson.dumps = partial(http_json.dumps, cls=JsonCommonEncoder, indent=4)
        self._url = url

    def post(self, header=None, data=None, json=None):
        response = requests.post(url=self._url, headers=header, data=data, json=json)
        if response.status_code == 200:
            result = response.json()
        else:
            result = response.text

        return result
