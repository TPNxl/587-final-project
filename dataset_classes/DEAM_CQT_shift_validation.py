import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from dataset_classes.DEAM_CQT_circshift import DEAM_CQT_Dataset_With_CircShift

class DEAM_CQT_Shift_Validation_Set(DEAM_CQT_Dataset_With_CircShift):
    def __init__(self, annot_path: str, audio_path: str, save_files: bool, transform_path: str, transform_name: str, transform_func=librosa.feature.chroma_cqt, start_s=15, dur=30, train=True):
        super().__init__(annot_path, audio_path, save_files, transform_path, transform_name, transform_func, start_s, dur, train)

        self.test_transf = super().calculate_transform(index=1, save_files=False)
        self.transform_width = self.test_transf.shape[1]
        self.int_sz = super().internal_size()

    def __len__(self):
        return self.int_sz

    def calculate_transform(self, index: int, save_files = False):
        out = torch.empty((self.transform_width, *self.test_transf.shape), dtype=self.test_transf.dtype, device=self.test_transf.device)
        for i in range(self.transform_width):
            out[i] = super().calculate_transform(index + self.int_sz * i, save_files=save_files)
        return out    
    
    def get_annots(self, index: int, len: int):
        annots = super().get_annots(index, len)
        out = torch.empty((self.transform_width, *annots.shape), dtype=annots.dtype, device=annots.device)
        for i in range(self.transform_width):
            out[i] = annots
        return out    

    def __getitem__(self, index: int):
        chroma_cq = self.calculate_transform(index, save_files=self.save_files)
        annots = self.get_annots(index, chroma_cq[0].shape[0])

        return chroma_cq, annots
