import torch
from collections import OrderedDict
import os
import sys

def extract_backbone(state_dict):
    keys_to_delete = []
    new_dict = OrderedDict()
    for key in state_dict:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = state_dict[key]
    
    for key in new_dict:
        print(key)
    
    return new_dict


def main(argv):
    chkpt = torch.load(argv[1])
    new_sd = extract_backbone(chkpt['state_dict'])
    new_filename = os.path.splitext(argv[1])[0] + '_backbone' + os.path.splitext(argv[1])[1]
    print(new_filename)
    torch.save({'state_dict': new_sd}, new_filename)

if __name__ == '__main__':
    main(sys.argv)