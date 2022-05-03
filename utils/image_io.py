# @Author  : zhangyouyuan (zyyzhang@fuzhi.ai)
# @Desc    :
import numpy as np
import imageio
import cv2
import os
import base64

from PIL import Image
from io import BytesIO

SUPPORT_IMAGE_FORMAT = {'.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP'}
SUPPORT_VIDEO = {'.mp4', '.avi'}

__all__ = [
    'read_image',
    'encode_image_base64',
    'encode_image_base64_from_file',
    'decode_base64_image',
    'save_image',
    'read_video',
    'encode_video_base64',
    'encode_video_base64_from_file',
    'decode_base64_video',
    'save_video'
]


def read_image(path):
    """
    以二进制的形式读入并解码，解决中文及大多数图像读取问题
    :param path:
    :return:
    """
    # 读入 bin
    with open(path, 'rb') as f:
        image_bin = f.read()

    # bin 转 bin
    image = Image.open(BytesIO(image_bin))
    image = np.array(image)

    return image


def encode_image_base64(image, to_str=False):
    """
    将图片编码成base64
    :param image:
    :param to_str: 是否需要将bytes转成str
    :return:
    """
    image = np.array(image)
    if image.shape.__len__() == 3:
        image = image[:, :, ::-1]  # 为了确保cv2编码后与输入颜色顺序一致
    image_str = cv2.imencode('.png', image)[1].tobytes()
    image_base64 = base64.b64encode(image_str)

    if to_str:
        image_base64 = str(image_base64, encoding='utf-8')
    return image_base64


def encode_image_base64_from_file(path, to_str=False):
    """
    从文件读取图片并编码成base64
    :param path:
    :param to_str: 是否将bytes转成str
    :return:
    """
    image = read_image(path)
    image_base64 = encode_image_base64(image)

    if to_str:
        image_base64 = str(image_base64, encoding='utf-8')
    return image_base64


def decode_base64_image(image_base64):
    """
    将base64的image解码成image
    :param image_base64:
    :return:
    """
    image_encode_bin = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_encode_bin))
    image = np.array(image)
    return image


def save_image(image, path):
    """
    保存图片
    :param image:
    :param path:
    :return:
    """
    folder_name = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    imageio.imsave(path, image, quality=100)
    print(f'success save image to {path}')


def read_video(video_base64_or_path, need_fps=False):
    """
    读取视频
    :param video_base64_or_path:
    :param need_fps: 是否需要返回帧
    :return:
    """
    if isinstance(video_base64_or_path, bytes):
        video_bin = base64.b64decode(video_base64_or_path)
        video_capture = imageio.get_reader(video_bin)
    else:
        video_capture = imageio.get_reader(video_base64_or_path)

    # 读取视频
    fps = video_capture.get_meta_data()['fps']
    frames = list(video_capture.iter_data())

    print('read video succeed ...')
    return (frames, fps) if need_fps else frames


def encode_video_base64(images, fps=25, to_str=False):
    """
    是视频编码成base64
    :param images: 视频帧组
    :param fps: 指定帧率
    :param to_str: 是否将bytes转成str
    :return:
    """
    cur_path = os.path.dirname(os.path.abspath(__file__))
    path = f'{cur_path}/infer_visual_video.mp4'
    save_video(images, path, fps=fps)
    video_base64 = encode_video_base64_from_file(path)
    os.remove(path)

    if to_str:
        video_base64 = str(video_base64, encoding='utf-8')
    return video_base64


def encode_video_base64_from_file(video_path, to_str=False):
    """
    从文件读取视频，并编码成base64
    :param video_path:
    :param to_str:  是否将bytes转成str
    :return:
    """
    with open(video_path, "rb") as f_video:
        video_bytes = f_video.read()
        video_base64 = base64.b64encode(video_bytes)
        f_video.close()

    if to_str:
        video_base64 = str(video_base64, encoding='utf-8')
    return video_base64


def decode_base64_video(video, need_fps=False):
    """
    解码视频
    :param video:
    :param need_fps: 是否需要返回帧率
    :return:
    """
    video = base64.b64decode(video)
    video_list = []
    fps = 0
    for suffix in SUPPORT_VIDEO:
        try:
            video_reader = imageio.get_reader(video, format=suffix[1:])
            fps = video_reader.get_meta_data()['fps']
            video_list = []
            for img in video_reader:
                video_list.append(img)
            break
        except Exception as exp:
            video_list = []
            fps = 0
            continue

    return (video_list, fps) if need_fps else video_list


def save_video(images, path, fps=25, size_w_h=None):
    """
    保存视频
    :param images:
    :param path:
    :param fps: 帧率
    :param size_w_h: 可指定视频的宽高
    :return:
    """
    folder_name = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    size_w_h = size_w_h if size_w_h else np.array(images[0]).shape[:2][::-1]
    video_writer = cv2.VideoWriter(path, fourcc, fps, size_w_h)

    images = np.array(images, dtype=np.uint8)
    if images.shape[-1] == 3:
        images = images[:, :, :, ::-1]  # cv2作妖
    for frame in images:
        video_writer.write(frame)
    video_writer.release()
    print(f'succeed save video to {path}')


def debug():
    image_path = r'1.jpeg'
    image = read_image(image_path)
    image_base64 = encode_image_base64(image)
    image_base64 = encode_image_base64_from_file(image_path)
    image = decode_base64_image(image_base64)
    save_image(image, 'new_' + image_path)

    video_path = r'1.avi'
    video = read_video(video_path)
    video_base64 = encode_video_base64(video)
    video_base64 = encode_video_base64_from_file(video_path)
    video = decode_base64_video(video_base64)
    save_video(video, 'new_' + video_path)


if __name__ == '__main__':
    debug()
