import torch
from collections import OrderedDict
import os
import sys

def extract_backbone(state_dict):
    keys_to_delete = []
    new_dict = OrderedDict()
    for key in state_dict:
        if key.startswith('backbone.'):
            #new_dict[key[9:]] = state_dict[key]
            #print(key)
            if ('body'+key[8:]) in state_dict:
                if not torch.equal(state_dict[key], state_dict['body'+key[8:]]):
                    print(key, 'is not equal to', 'body'+key[8:])
            else:
                print(key, 'in backbone but not in body')
        if key.startswith('body.'):
            new_dict[key[5:]] = state_dict[key]
            print(key)
            #assert(torch.equal(new_dict[key[9:]], state_dict['body'+key[8:]]))
    
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