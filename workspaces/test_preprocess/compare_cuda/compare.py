import numpy as np
import cv2
import os
import time

def load_tensor(file):
    
    data = open(file, "rb").read()
    header = np.frombuffer(data[:16 * 4], dtype=np.uint32)
    magic_number = header[0]
    assert magic_number == 0xFFCC1122, f"Invalid tensor file."
    
    dtype_map = {
        0: np.uint8,
        1: np.float32
    }
    nptype = dtype_map[header[1]]
    ndims  = header[2]
    dims   = header[3:3+ndims]
    return np.frombuffer(data[16*4:], dtype=nptype).reshape(*dims)
    
    
root = "../CUDA_Result"
files = os.listdir(root)
files = [file.split("_ori.tensor")[0] for file in files if file.endswith("_ori.tensor")]
print(files)

total_time = 0.

for file in files:
    ori = load_tensor(f"{root}/{file}_ori.tensor")
    cpp = load_tensor(f"{root}/{file}_cuda.tensor")
    
    start_time = time.time() # 起始
    resized = cv2.resize(ori, dsize=None, fx=0.48, fy=0.48) # 双线性插值
    croped  = resized[176:, 32:-32] # 裁剪
    end_time = time.time()  #  终止
    total_time += end_time - start_time  # 累加最终得到总耗时
    
    diff = np.abs(croped.astype(np.float32) - cpp)
    print(diff.max())
    cv2.imwrite(f"diff_{file}.png", diff * 255)
    cv2.imwrite(f"ori_{file}.jpg", ori)
    cv2.imwrite(f"croped_{file}.jpg", croped)
    cv2.imwrite(f"cuda_{file}.jpg", cpp)
    # break
print(total_time)