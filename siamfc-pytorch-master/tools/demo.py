from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc1 import TrackerSiamFC


if __name__ == '__main__':
    seq_dir = os.path.expanduser(r'G:\my_data\PTB-TIR\tirsequences_new2\tirsequences\airplane')
    img_files = sorted(glob.glob(seq_dir + '\img\*.jpg'))

    # anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')
    anno = np.loadtxt(r'F:\program\siamfc-pytorch-dmeo\siamfc-pytorch-master\tools\results\OTBPTB-TIR\SiamFC\airplane.txt',delimiter=',')
    net_path = 'pretrained/siamfc_alexnet_e49.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track(img_files, anno[0], visualize=True)
