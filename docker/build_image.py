
import os
import shutil
from config import image_name, dockerfile_name, image_tar_save_path

# 移除原有的镜像
if os.path.exists(image_tar_save_path):
    shutil.rmtree(image_tar_save_path)

os.system(f'sudo docker build -t {image_name} -f {dockerfile_name} --no-cache ../')  # 不要忘记最后的一个或者两个点
