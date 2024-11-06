import os

# 指定要读取的文件夹路径
folder_path = '/icislab/volume1/lxc/TCMatting/data/am2k/train/alpha'  # 替换为你的文件夹路径
# 指定要保存的txt文件路径
output_file = '/icislab/volume1/lxc/TCMatting/data/am2k/train/filesname.txt'

# 获取文件夹中的所有文件
image_names = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# 将图片名称写入txt文件
with open(output_file, 'w') as file:
    for name in image_names:
        file.write(name + '\n')

print(f'已保存 {len(image_names)} 个图片名称到 {output_file}')
