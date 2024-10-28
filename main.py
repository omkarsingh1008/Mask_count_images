import cv2
import numpy as np
import os
import threading

import glob
import logging

logging.basicConfig(filename="mask_pixel_count.log", level=logging.INFO, format="%(asctime)s - %(message)s")


input_dir = "/home/omkar/Downloads/Online-test"
output_dir = "/home/omkar/Downloads/Online-test/mask"
os.makedirs(output_dir, exist_ok=True)

total_mask=0

def count_mask_image(file_path):
    global total_mask
    
    image = cv2.imread(file_path)

    mask = np.all( image > 200, axis=2).astype(np.uint8) * 255
    
    masked_pixel_count = np.sum(mask == 255)
    
    filename = os.path.basename(file_path)
    mask_filename = os.path.join(output_dir, f"mask_{filename}")
    cv2.imwrite(mask_filename, mask)
    total_mask+=masked_pixel_count
    return masked_pixel_count


image_files = glob.glob(os.path.join(input_dir, "*.jpg"))


thead_list = []

for image_apth in image_files:
    print(image_apth)
    thread = threading.Thread(target=count_mask_image, args=[image_apth])
    thread.start()
    thead_list.append(thread)


for thread in thead_list:
    thread.join()



logging.info(f"Total number of pixels with max mask value across all images: {total_mask}")
print(f"Total number of pixels with max mask value across all images: {total_mask}")
