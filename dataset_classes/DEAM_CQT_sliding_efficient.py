import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from dataset_classes.DEAM_CQT import *

def display_cqt(chroma_cq, hop_length):
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    img = librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time',hop_length=hop_length)
    ax.set(title='chroma_cqt')
    fig.colorbar(img, ax=ax)

class DEAM_CQT_Dataset_Sliding_Efficient(DEAM_CQT_Dataset):
    def __init__(self, annot_path: str, audio_path: str, save_files: bool, transform_path: str, transform_name: str, transform_func=librosa.feature.chroma_cqt, start_s=15, dur=30, sr=None, train=True, 
                 use_transform_array = True, use_annots_array = True, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        super(DEAM_CQT_Dataset_Sliding_Efficient, self).__init__(annot_path, audio_path, save_files, transform_path, transform_name, transform_func, start_s, dur, train)

        self.LEN_WINDOW = 10
        self.num_perms = self.dur * 2 - self.LEN_WINDOW + 1
        self.internal_size = super().__len__()

        test_transf = self.calculate_transform(0, 0, self.LEN_WINDOW, save_files=False, instant=True)
        self.transform_width = test_transf.shape[1]

        song_id = self.annot_df.loc[1, "song_id"]
        song_path = "".join([self.audio_path, "/", str(song_id), ".mp3"])
        _, sr = librosa.load(song_path)
        self.sr = sr

        self.device = device

        self.use_transform_array = use_transform_array
        self.transform_array = None

        self.use_annots_array = use_annots_array
        self.annots_array = None
        

    def __len__(self):
        return self.internal_size * self.num_perms
    
    def calculate_transform(self, index: int, start: int, end: int, save_files = False, instant=False):
        path = self.get_path(index)
        if os.path.exists(path) and not instant:
            chroma_cq = torch.load(path)
        else:
            song_id = self.annot_df.loc[index, "song_id"]
            song_path = "".join([self.audio_path, "/", str(song_id), ".mp3"])
            y, sr = librosa.load(song_path)
            assert sr == self.sr
            HOP_LENGTH = int(sr/2) + 1
            y = y[self.start_s*sr:(self.start_s + self.dur)*sr]
            chroma_cq = self.transform_func(y=y, sr=sr, hop_length=HOP_LENGTH)
            chroma_cq = torch.tensor(chroma_cq)
            chroma_cq = torch.transpose(chroma_cq, 0, 1)
            if save_files:
                torch.save(chroma_cq, path)

        chroma_cq = chroma_cq[start:end]
        return chroma_cq
    
    def set_up_transform_array(self, replace=False):
            if self.transform_array is None or replace is True:
                chroma_cq = self.calculate_transform(0, 0, self.LEN_WINDOW, save_files=self.save_files)
                self.transform_array = torch.empty((self.__len__(), *chroma_cq.shape), dtype=chroma_cq.dtype, device=self.device)
                for i in range(self.__len__()):
                    file_num = int(np.floor(i / self.num_perms))
                    start = i % self.num_perms
                    end = start + self.LEN_WINDOW
                    chroma_cq = self.calculate_transform(file_num, start, end, save_files=self.save_files)
                    chroma_cq = chroma_cq.to(self.device)
                    self.transform_array[i] = chroma_cq
                self.transform_array = self.transform_array.float()
                self.transform_array = self.transform_array.to(self.device)

    def set_up_annots_array(self):
        self.annots_array = torch.tensor(self.annot_df.to_numpy())
        self.annots_array = self.annots_array.float()
        self.annots_array = self.annots_array.to(self.device)
    
    def get_annots(self, index: int, start: int, end: int, len: int):
        annots = self.annot_df.loc[index].to_numpy()
        annots = annots[2 + start : 2 + end]
        annots = torch.tensor(annots)
        return annots
    
    def get_annots_from_array(self, index: int, start: int, end: int, len: int):
        annots = self.annots_array[index]
        annots = annots[2 + start : 2 + end]
        return annots
    
    def __getitem__(self, index: int):
        file_num = int(np.floor(index / self.num_perms))
        start = index % self.num_perms
        end = start + self.LEN_WINDOW
        if self.transform_array is not None:
            chroma_cq = self.transform_array[index]
        elif self.use_transform_array:
            self.set_up_transform_array()
            chroma_cq = self.transform_array[index]
        else:
            # print("File_num: %d\nStart: %d\nEnd: %d" % (file_num, start, end))
            chroma_cq = self.calculate_transform(file_num, start, end, save_files=self.save_files)
        if self.annots_array is not None:
            annots = self.get_annots_from_array(file_num, start, end, chroma_cq.shape[0])
        elif self.use_annots_array:
            self.set_up_annots_array()
            annots = self.get_annots_from_array(file_num, start, end, chroma_cq.shape[0])
        else:
            annots = self.get_annots(file_num, start, end, chroma_cq.shape[0])

        # print(chroma_cq.type())
        # print(annots.type())

        return chroma_cq, annots