"""
Creates delta feature ops 

Author: Deblin Bagchi
Date: Fall 2017
"""

import tensorflow as tf
import numpy as np
import random
import os
import struct
 
from data_io import smart_open 
#assert order >=0 && order <=2 , "invalid order specified"
#assert window >=0 && window < 1000, "invalid window specified"
order=2
window=2        
scales = []
scdel1 = np.ones((1))
scales.append(scdel1)
for i in range(1,order+1): 
    prev_scales = scales[i-1]
    _window = window; 
    assert _window!=0, "invalid window size" 
    prev_offset = int((prev_scales.shape[0]-1)/2)
    cur_offset = prev_offset + _window;
    scdel = np.zeros(shape=(prev_scales.shape[0] + 2*window,), dtype=np.float)
    scales.append(scdel)
    cur_scales = scales[i]
   
    normalizer = 0.0;
    for j in range(-window,window+1):
        normalizer += j*j;
        for k in range(-prev_offset,prev_offset+1):
            cur_scales[j+k+cur_offset] += j * prev_scales[k+prev_offset]
    scaler = float(1/normalizer)
    cur_scales = scaler * cur_scales
    scales[i] = cur_scales

def Process(input_feats,frame):
    num_frames = input_feats.shape[0]
    feat_dim = input_feats.shape[1]
    output_frame = []

    for i in range(0,order+1):
        scale = scales[i]
        max_offset = int((scale.shape[0] - 1) / 2)
        output = np.zeros(feat_dim) 
        for j in range(-max_offset, max_offset+1):
            offset_frame = frame + j
            if (offset_frame < 0):
                offset_frame = 0
            elif (offset_frame >= num_frames):
                offset_frame = num_frames - 1
            sscale = scale[j + max_offset]
            if (sscale != 0):
                output = output + (sscale * input_feats[offset_frame][:])
        output_frame.append(output)
    result = np.stack(output_frame).flatten()
    return result

def process_data(ark_base_dir=os.getcwd(), scp_fn="data-fbank/train_si84_noisy/feats.scp"):
    ark_dict = {}
    totframes = 0
    lines = 0
    with open(scp_fn) as f:
        for line in f:
            lines = lines + 1
            if lines>10:
                break
            if line == "":
                continue
            utt_id, path_pos = line.replace("\n", "").split()
            ark_path, pos = path_pos.split(":")
            ark_path = os.path.join(ark_base_dir, ark_path)
            ark_read_buffer = smart_open(ark_path, "rb")
            ark_read_buffer.seek(int(pos),0)
            header = struct.unpack("<xcccc", ark_read_buffer.read(5))
            #assert header[0] == "B", "Input .ark file is not binary"
            rows = 0
            cols = 0
            m,rows = struct.unpack("<bi", ark_read_buffer.read(5))
            n,cols = struct.unpack("<bi", ark_read_buffer.read(5))
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows*cols*4), dtype=np.float32)
            utt_mat = np.reshape(tmp_mat, (rows, cols))
            #utt_mat_list=utt_mat.tolist()
            ark_read_buffer.close()
            ark_dict[utt_id] = utt_mat

    return ark_dict

ark_dict = process_data()
ark_delta_dict = process_data(scp_fn="data-fbank/train_si84_delta_noisy/feats.scp")
delta_ids = sorted(ark_delta_dict.keys())
ids = sorted(ark_dict.keys())
mats = np.vstack([ark_dict[i] for i in ids])
delta_mats = np.vstack([ark_delta_dict[i] for i in delta_ids])

print(mats.shape)
print(delta_mats.shape)
result = np.vstack([Process(mats,i) for i in range(mats.shape[0])])
print(result[0][:10])
print(result[0][40:50])
print(result[0][80:90])
print(delta_mats[0][:10])
print(delta_mats[0][40:50])
print(delta_mats[0][80:90])
print(result.shape)
