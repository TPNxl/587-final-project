import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from dataset_classes.DEAM_CQT_sliding_efficient import DEAM_CQT_Dataset_Sliding_Efficient

def display_cqt(chroma_cq, hop_length):
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    img = librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time',hop_length=hop_length)
    ax.set(title='chroma_cqt')
    fig.colorbar(img, ax=ax)

class DEAM_CQT_Dataset_With_CircShift_Sliding(torch.utils.data.Dataset):
    def __init__(self, annot_path: str, audio_path: str, save_files: bool, transform_path: str, transform_name: str, transform_func=librosa.feature.chroma_cqt, start_s=15, dur=30, train=True):
        self.parent_dataset = DEAM_CQT_Dataset_Sliding_Efficient(annot_path, audio_path, save_files, transform_path, transform_name, transform_func, start_s, dur, train)

        self.LEN_WINDOW = 10
        test_transf = self.parent_dataset.calculate_transform(0, 0, self.LEN_WINDOW, save_files=False, instant=True)
        self.transform_width = test_transf.shape[1]
        self.int_sz = self.parent_dataset.__len__()
        

    def __len__(self):
        return self.int_sz * self.transform_width
    

    def __getitem__(self, index: int):
        # print("int_sz: ", self.int_sz2)
        new_index = index % self.int_sz
        # print("New index: ", new_index)
        roll_val = int(np.floor(index / self.int_sz))
        # # print("Roll val: ", roll_val)
        # file_num = int(np.floor(new_index / self.num_perms))
        # start = new_index % self.num_perms
        # end = start + self.LEN_WINDOW
        # # print("File_num: %d\nStart: %d\nEnd: %d" % (file_num, start, end))
        # # chroma_cq = super().calculate_transform(file_num, start, end, save_files=self.save_files)
        
        # # annots = super().get_annots(file_num, start, end, chroma_cq.shape[0])

        # if self.transform_array is not None:
        #     chroma_cq = self.transform_array[index]
        # elif self.use_transform_array:
        #     self.set_up_transform_array()
        #     chroma_cq = self.transform_array[index]
        # else:
        #     # print("File_num: %d\nStart: %d\nEnd: %d" % (file_num, start, end))
        #     chroma_cq = self.calculate_transform(file_num, start, end, save_files=self.save_files)
        # if self.annots_array is not None:
        #     annots = self.get_annots_from_array(file_num, start, end, chroma_cq.shape[0])
        # elif self.use_annots_array:
        #     self.set_up_annots_array()
        #     annots = self.get_annots_from_array(file_num, start, end, chroma_cq.shape[0])
        # else:
        #     annots = self.get_annots(file_num, start, end, chroma_cq.shape[0])

        chroma_cq, annots = self.parent_dataset.__getitem__(new_index)
        
        chroma_cq = chroma_cq.roll(roll_val, dims=(1))

        return chroma_cq, annots
