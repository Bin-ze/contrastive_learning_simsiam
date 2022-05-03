# @Author  : zhangyouyuan (zyyzhang@fuzhi.ai)
# @Desc    :

import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
import os

from PIL import Image, ImageDraw

try:
    from utils import image_io
except Exception as exp:
    import image_io

plt.ion()
KEEP_DIGITS_NUM = 4

__all__ = [
    'get_line_visual_data',
    'get_mask_visual_data',
    'get_localed_visual_data',
    'get_visual_result',
    'trans_box_to_line',
    'create_not_exist_path',
    'draw_lines',
    'color_mask',
    'predict_debug'
]


def get_line_visual_data(infer_data, output_type='image'):
    """
    获取连线类visual_data
    :param infer_data:
    :param output_type:
    :return:
    """
    assert output_type in ['image', 'video']

    if 'image' in output_type:
        lines = trans_box_to_line(infer_data)
    elif 'video' in output_type:
        lines = []
        for frame_id, box_score_label in enumerate(infer_data):
            line = trans_box_to_line(box_score_label)
            lines.append({
                'frame_id': frame_id,
                'lines': line
            })
    else:
        raise ValueError(f'output_type must contain "image" or "video", but get {output_type}')

    visual_data = {
        'output_type': output_type,
        'annotation_type': 'line',
        'data': lines
    }
    return visual_data


def trans_box_to_line(box_score_label):
    """
    将检测结果转成visual_data，
    :param box_score_label: 每个元素为[x1, y1, x2, y2, score, 'category']
    :return:
    """
    lines = []
    for *bbox, score, label in box_score_label:
        points = [
            {'coordinate': [bbox[0], bbox[1]]},
            {'coordinate': [bbox[2], bbox[1]]},
            {'coordinate': [bbox[2], bbox[3]]},
            {'coordinate': [bbox[0], bbox[3]]},
            {'coordinate': [bbox[0], bbox[1]]},
        ]
        attribute = {
            'text': f'{label}_{round(score, KEEP_DIGITS_NUM)}',  # 显示的文本
        }
        lines.append({
            'points': points,
            'attribute': attribute
        })

    return lines


def get_mask_visual_data(infer_data, output_type='image'):
    """
    将mask组装成visual_data
    :param infer_data:
    :param output_type:
    :return:
    """
    assert output_type in ['image', 'video']

    if 'image' in output_type:
        if isinstance(infer_data, (list, np.ndarray)):
            mask = image_io.encode_image_base64(infer_data, to_str=True)
        else:
            mask = infer_data
    elif 'video' in output_type:
        if isinstance(infer_data, (str, bytes)):
            infer_data = image_io.decode_base64_video(infer_data)
        mask = []
        for frame_id, mask_i in enumerate(infer_data):
            mask.append({
                'frame_id': frame_id,
                'lines': image_io.encode_image_base64(mask_i, to_str=True)
            })
    else:
        raise ValueError(f'output_type must contain "image" or "video", but get {output_type}')

    visual_data = {
        'output_type': output_type,
        'annotation_type': 'mask',
        'data': mask
    }
    return visual_data


def get_localed_visual_data(infer_data, output_type='text'):
    """
    将图片、视频、文本等无需做改变的内容，组装成visual_data格式
    :param infer_data:
    :param output_type:
    :return:
    """
    if 'image' in output_type:
        if isinstance(infer_data, (list, np.ndarray)):
            infer_data = image_io.encode_image_base64(infer_data, to_str=True)
    if 'text' in output_type:
        if isinstance(infer_data, (list, str)):
            infer_data = infer_data[0] + '_' + infer_data[1]
    elif 'video' in output_type:
        if isinstance(infer_data, (list, np.ndarray)):
            infer_data = image_io.encode_video_base64(infer_data, to_str=True)
    visual_data = {
        'output_type': output_type,
        'annotation_type': None,
        'data': infer_data
    }
    return visual_data


def get_visual_result(visual_data, image_video_text):
    """
    根据visual_data获取可视化结果
    :param visual_data:
    :param image_video_text: 原始的图片、视频、或文本
    :return:
    """
    assert visual_data['output_type'] in ['image', 'video', 'text'], \
        f"annotation_type must be in ['image', 'video', 'text]"
    assert visual_data['annotation_type'] in ['line', 'mask', None], \
        f"annotation_type must be in ['line', 'mask', None]"

    if not visual_data['annotation_type']:
        return visual_data['data']

    visual_result = None
    if visual_data['annotation_type'] == 'line':
        if visual_data['output_type'] == 'image':
            image = image_video_text
            visual_result = draw_lines(image, visual_data['data'])
            visual_result = image_io.encode_image_base64(visual_result, to_str=True)

        elif visual_data['output_type'] == 'video':
            video = image_video_text
            visual_frames = []
            for frame in visual_data['data']:
                frame_id = frame['frame_id']
                lines = frame['lines']
                visual_frame = draw_lines(video[frame_id], lines)
                visual_frames.append(visual_frame)
            visual_result = image_io.encode_video_base64(visual_frames, fps=25, to_str=True)

    elif visual_data['annotation_type'] == 'mask':
        if visual_data['output_type'] == 'image':
            mask = visual_data['data']
            mask = image_io.decode_base64_image(mask)
            mask_color = color_mask(mask)
            visual_result = image_io.encode_image_base64(mask_color, to_str=True)
        elif visual_data['output_type'] == 'video':
            visual_frames = []
            for frame in visual_data['data']:
                frame_id = frame['frame_id']
                mask = image_io.decode_base64_image(frame['mask'])
                mask_color = color_mask(mask)
                visual_frames.append(mask_color)
            visual_result = image_io.encode_video_base64(visual_frames, fps=25, to_str=True)

    return visual_result


def create_not_exist_path(file_path):
    """
    输入可为path或者文件
    因此，如果是path，如果文件夹名包含点，则必须跟上斜杠/
    :param file_path:
    :return:
    """
    if not os.path.exists(file_path):
        file_name = os.path.basename(file_path)
        if len(file_name.split('.')) > 1 and not file_path.endswith('/'):  # if file or dir ?
            file_path = os.path.abspath(os.path.dirname(file_path))
        print("path: {} not exist, to create it".format(file_path))
        os.makedirs(file_path, exist_ok=True)


def draw_lines(image, lines):
    """
    根据visual_data的line画线
    :param image: 图片
    :param lines:
    :return:
    """
    image = np.array(image)
    image = Image.fromarray(image)
    image_draw = ImageDraw.Draw(image)
    width = 3

    for line in lines:
        points = line['points']
        attribute = line['attribute']
        for i in range(len(points) - 1):
            line_i = [points[i]['coordinate'], points[i + 1]['coordinate']]
            color = points[i].get('color', attribute.get('color', [255, 0, 0]))
            image_draw.line((*line_i[0], *line_i[1]), tuple(color), width=width)
    image_lines = np.array(image)
    return image_lines


def color_mask(mask):
    """
    给mask上色，score越低越蓝，score越高越红
    :param mask:
    :return:
    """
    # get color_sequence
    color_sequence = [[x, 255 - int(x / 4), 255 - x] for x in range(120, 0, -1)]  # 根据实际效果调参数-HSV
    color_sequence = np.array(color_sequence, dtype=np.uint8).reshape((1, -1, 3))
    color_sequence = cv2.cvtColor(color_sequence, cv2.COLOR_HSV2RGB)
    color_sequence = np.reshape(color_sequence, newshape=(-1, 3))

    # color
    min_value, max_value = np.min(mask), np.max(mask)
    get_color = lambda x: color_sequence[int((x - min_value) * (len(color_sequence) - 1) / (max_value - min_value))]
    mask_color = np.zeros(shape=(*mask.shape, 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask_color[i, j] = get_color(mask[i, j])

    return mask_color


def predict_debug(content, image_type='image', need_visual_data=False, need_visual_result=False):
    """
    HTTP: /autotable/predict POST
    """
    predicts = []
    if isinstance(content, (list, tuple)):
        for image in content:
            predicts.append(infer_single_sample(image, image_type, need_visual_data, need_visual_result))
    else:
        predicts.append(infer_single_sample(content, image_type, need_visual_data, need_visual_result))

    return predicts


def infer_single_sample(content, image_type, need_visual_data, need_visual_result):
    video = image_io.decode_base64_image(content)
    mask = np.zeros(shape=(200, 300), dtype=np.uint8)
    mask[23: 56, 34: 67] = 1
    mask[100: 134, 134: 167] = 2
    mask[150: 200, 134: 167] = 3
    infer_data = mask

    # 可视化
    if need_visual_data or need_visual_result:
        # video-None
        visual_data = get_mask_visual_data(infer_data, output_type='image')

        if need_visual_result:
            visual_result = get_visual_result(visual_data, video)
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


def main_func():
    # image
    image = imageio.imread('1.jpeg')
    image_encode = image_io.encode_image_base64(image)
    infer_param = dict(
        content=image_encode,
        image_type='image',
        need_visual_data=True,
        need_visual_result=True
    )

    # infer
    infer_result = predict_debug(**infer_param)
    visual_result = infer_result[0]['visual_result']
    if isinstance(visual_result, (bytes, str)):
        plt.imshow(image_io.decode_base64_image(visual_result))
        plt.waitforbuttonpress()
    else:
        print(visual_result)


if __name__ == '__main__':
    main_func()
