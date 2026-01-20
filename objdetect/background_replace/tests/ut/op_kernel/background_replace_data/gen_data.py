import os
import sys
import numpy as np


def gen_golden_data_simple(dtype, channels):
    if(channels == '1'):
        bkg = np.random.uniform(0, 15, (480, 640, 1)).astype(dtype)
        src = np.random.uniform(0, 15, (480, 640, 1)).astype(dtype)
        mask = np.random.uniform(0, 15, (480, 640, 1)).astype(np.float16)
        out = (bkg - bkg * mask + src * mask).astype(dtype)
    else:
        bkg = np.random.uniform(0, 15, (480, 640, 3)).astype(dtype)
        src = np.random.uniform(0, 15, (480, 640, 3)).astype(dtype)
        mask = np.random.uniform(0, 15, (480, 640, 1)).astype(np.float16)
        out = (bkg - bkg * mask + src * mask).astype(dtype)
    bkg.tofile("./bkg.bin")
    src.tofile("./src.bin")
    mask.tofile("./mask.bin")
    out.tofile("./golden.bin")

if __name__ == "__main__":
    datatype = sys.argv[1]
    if datatype == "uint8":
        dtype = np.uint8
    else:
        dtype = np.float16
    gen_golden_data_simple(dtype, sys.argv[2])