from __future__ import absolute_import

import os
from got10k.experiments import *

from multi_AlexNet.siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained_GAT/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir = os.path.expanduser(r'G:\my_data\PTB-TIR\tirsequences_new2\tirsequences')
    e = ExperimentOTB(root_dir, version='PTB-TIR')
    e.run(tracker)
    e.report([tracker.name])

