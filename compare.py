import numpy as np
import os 
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def compare_two_tensor(output, output_c):
    if len(output) != len(output_c):
        print("Tensor size is not same : output_py=%d  output_c=%d" % (len(output), len(output_c)))
        exit()

    max_idx = -1
    max_diff = 0
    cnt_diff = 0
    r0 = 0
    r = 0
    for idx in range(len(output)):
        if output[idx] != output_c[idx]:
            cnt_diff += 1
            diff = abs(output[idx] - output_c[idx])
            if diff > max_diff:
                max_diff = diff
                max_idx = idx
                r0 = output[idx]
                r = output_c[idx]
            print("%6d output_py=%10.6f output_c=%10.6f diff=%.6f" % (idx, output[idx], output_c[idx], diff))
    if max_diff > 0:
        print("cnt_total=%d cnt_diff=%d max_idx=%6d output_py=%10.6f output_c=%10.6f diff=%.6f" % (
        len(output), cnt_diff, max_idx, r0, r, max_diff))
    else:
        print("All data is same!")


output_c = np.fromfile(f"{CUR_DIR}/x", dtype=np.float32)
output_py = np.fromfile(f"{CUR_DIR}/x_pt", dtype=np.float32)
compare_two_tensor(output_py, output_c)
