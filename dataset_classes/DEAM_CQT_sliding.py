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

class DEAM_CQT_Dataset_Sliding(DEAM_CQT_Dataset):
    def __init__(self, annot_path: str, audio_path: str, save_files: bool, transform_path: str, transform_name: str, transform_func=librosa.feature.chroma_cqt, start_s=15, dur=30, sr=None, train=True):
        super().__init__(annot_path, audio_path, save_files, transform_path, transform_name, transform_func, start_s, dur, train)

        self.LEN_WINDOW = 10
        self.num_perms = self.dur * 2 - self.LEN_WINDOW + 1
        self.internal_size = super().__len__()

        song_id = self.annot_df.loc[1, "song_id"]
        song_path = "".join([self.audio_path, "/", str(song_id), ".mp3"])
        _, sr = librosa.load(song_path)
        self.sr = sr

    def __len__(self):
        return self.internal_size * self.num_perms
    
    def calculate_transform(self, index: int, start: int, end: int, save_files = False):
        path = self.get_path(index)
        if os.path.exists(path):
            chroma_cq = torch.load(path)
        else:
            song_id = self.annot_df.loc[index, "song_id"]
            song_path = "".join([self.audio_path, "/", str(song_id), ".mp3"])
            y, sr = librosa.load(song_path)
            assert sr == self.sr
            HOP_LENGTH = int(sr/2) + 1
            y = y[self.start_s*sr:(self.start_s + self.dur)*sr]
            chroma_cq = self.transform_func(y=y, sr=sr, hop_length=HOP_LENGTH)
            chroma_cq = torch.tensor(chroma_cq).float()
            chroma_cq = torch.transpose(chroma_cq, 0, 1)
            if save_files:
                torch.save(chroma_cq, path)

        chroma_cq = chroma_cq[start:end]
        return chroma_cq
    
    def get_annots(self, index: int, start: int, end: int, len: int):
        annots = self.annot_df.loc[index].to_numpy()
        annots = annots[2 + start : 2 + end]
        annots = torch.tensor(annots).float()
        return annots

    def __getitem__(self, index: int):
        file_num = int(np.floor(index / self.num_perms))
        start = index % self.num_perms
        end = start + self.LEN_WINDOW
        # print("File_num: %d\nStart: %d\nEnd: %d" % (file_num, start, end))
        chroma_cq = self.calculate_transform(file_num, start, end, save_files=self.save_files)
        annots = self.get_annots(file_num, start, end, chroma_cq.shape[0])

        return chroma_cq, annots