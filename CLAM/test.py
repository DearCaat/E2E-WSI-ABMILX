import time
import os
import psutil
import cv2
import numpy as np
from PIL import Image
import io
import concurrent.futures
import glob
import pickle
import torch
from torchvision.transforms import v2

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 返回以MB为单位的内存使用量

def read_and_decode_pil(byte_data):
    img_data = pickle.loads(byte_data)
    to_tensor = v2.ToImage()
    with Image.open(io.BytesIO(img_data)) as img:
        return to_tensor(img)

def read_and_decode_cv2(byte_data):
    img_data = pickle.loads(byte_data)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

def run_pil_test(byte_data_list):
    for byte_data in byte_data_list:
        read_and_decode_pil(byte_data)

def run_cv2_test(byte_data_list):
    for byte_data in byte_data_list:
        read_and_decode_cv2(byte_data)

def prepare_byte_data(image_folder):
    byte_data_list = []
    for image_path in glob.glob(os.path.join(image_folder, "*.jpg")):
        with open(image_path, 'rb') as f:
            img_data = f.read()
        byte_data = pickle.dumps(img_data)
        byte_data_list.append(byte_data)
    return byte_data_list

def run_multithreaded_test(byte_data_list, num_threads, method):
    start_time = time.time()
    start_memory = get_memory_usage()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        chunk_size = len(byte_data_list) // num_threads
        chunks = [byte_data_list[i:i + chunk_size] for i in range(0, len(byte_data_list), chunk_size)]

        if method == 'pil':
            futures = [executor.submit(run_pil_test, chunk) for chunk in chunks]
        else:
            futures = [executor.submit(run_cv2_test, chunk) for chunk in chunks]

        concurrent.futures.wait(futures)

    end_time = time.time()
    end_memory = get_memory_usage()

    total_time = end_time - start_time
    total_memory = end_memory - start_memory

    return total_time, total_memory

def main(image_folder, num_threads):
    print(f"Preparing byte data from images in folder: {image_folder}")
    byte_data_list = prepare_byte_data(image_folder)
    print(f"Number of images: {len(byte_data_list)}")
    print(f"Number of threads: {num_threads}")

    # Test PIL method
    pil_time, pil_memory = run_multithreaded_test(byte_data_list, num_threads, 'pil')
    print(f"PIL method:")
    print(f"  Time taken: {pil_time:.4f} seconds")
    print(f"  Memory used: {pil_memory:.2f} MB")

    # Test OpenCV method
    cv2_time, cv2_memory = run_multithreaded_test(byte_data_list, num_threads, 'cv2')
    print(f"OpenCV method:")
    print(f"  Time taken: {cv2_time:.4f} seconds")
    print(f"  Memory used: {cv2_memory:.2f} MB")

    print(f"\nComparison:")
    print(f"  Time difference: PIL is {pil_time/cv2_time:.2f}x slower than OpenCV")
    print(f"  Memory difference: PIL uses {pil_memory/cv2_memory:.2f}x more memory than OpenCV")

if __name__ == "__main__":
    image_folder = "/XXXwsi_data/patch/brca/imgs/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291"  # 替换为包含测试图像的文件夹路径
    num_threads = 8  # 设置想要测试的线程数
    main(image_folder, num_threads)