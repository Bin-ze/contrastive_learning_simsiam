
# 当前机器的普通用户
user = 'deepwisdom'

# build 镜像
tag = 'v1.0'
image_prefix = 'frvsr_gan'
image_suffix = f'image'
image_name = f'ccr.ccs.tencentyun.com/deepwisdom/{image_prefix}_{image_suffix}:{tag}'
dockerfile_name = 'Dockerfile'

# 生成容器
container_name = f'{image_prefix}_test'
gpu_id = '0'  # nvidia-docker 指定gpu号

# 保存tar.gz
image_tar_save_path = 'image'
image_tar_save_name = image_name.replace(':', '_') + '.tar.gz'

# 推送时新tag
new_tag_name = f'ccr.ccs.tencentyun.com/deepwisdom/{image_prefix}_{image_suffix}:{tag}'
