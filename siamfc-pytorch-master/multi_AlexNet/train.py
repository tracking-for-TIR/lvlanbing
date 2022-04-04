from __future__ import absolute_import

import os
from got10k.datasets import *


from multi_AlexNet.siamfc import TrackerSiamFC


if __name__ == '__main__':
    root_dir = os.path.expanduser(r'G:\my_data\LSOTB-TIR\Training_Dataset\new\LSOTB-TIR_TrainingData\LSOTB-TIR_TrainingData\TrainingData')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    root_dir2 = os.path.expanduser(r'F:\data\GOT\train_data')
    seqs2 = GOT10k(root_dir2, subset="train", return_meta=True)

    for i in range(len(seqs.anno_files)):
        seqs.anno_files.append(seqs2.anno_files[i])
        seqs.seq_dirs.append(seqs2.seq_dirs[i])
        seqs.seq_names.append(seqs2.seq_names[i])

    tracker = TrackerSiamFC()
    tracker.train_over(seqs)
