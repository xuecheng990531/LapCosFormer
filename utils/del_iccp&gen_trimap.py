from PIL import Image
import cv2
import numpy as np
import os

maskdir='/icislab/volume1/lxc/TCMatting/data/dis646/train/img'
savedir='/icislab/volume1/lxc/TCMatting/data/am2k/train/trimap'
list='/icislab/volume1/lxc/TCMatting/data/am2k/train/filesname.txt'

def gen_trimap(alpha, ksize=3, iterations=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape) + 128
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap

def remove_icc_profile(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif')):
            file_path = os.path.join(directory, filename)
            try:
                with Image.open(file_path) as img:
                    # Remove ICC profile by saving without it
                    img.save(file_path, icc_profile=None)
                    print(f"Removed ICC profile from {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == '__main__':
    remove_icc_profile(maskdir)
    # with open(list, 'r') as f:
    #     for line in f:
    #         line = line.strip()
    #         alpha = cv2.imread(os.path.join(maskdir, line), cv2.IMREAD_GRAYSCALE)
    #         trimap = gen_trimap(alpha)
    #         cv2.imwrite(os.path.join(savedir, line), trimap)
    #         print(f"Saved {line}")
    # print("Done")