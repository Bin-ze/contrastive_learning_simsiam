import imageio
import os

from utils.service_tools import get_host_ip, HttpClient
from utils.image_io import encode_image_base64, decode_base64_image


def main_func():
    url = f'http://{get_host_ip()}:8080/autotable/predict'
    path = '/data/flower_data/valid/'
    for i in sorted(os.listdir(path)):
        print(i)
        for j in sorted(os.listdir(os.path.join(path, i))):
            image_path = os.path.join(path, i, j)
            image = imageio.imread(image_path)
            image = encode_image_base64(image)
            post_data = {'content': image,'need_visual_data': True, 'need_visual_result': True}
            client = HttpClient(url)
            result = client.post(json=post_data)
            print(result)

    print('finished ...')


if __name__ == '__main__':
    main_func()
