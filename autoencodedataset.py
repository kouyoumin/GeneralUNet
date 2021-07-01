import torch
from torch.utils.data import Dataset
import os
import pydicom
import re
import fnmatch
from pathlib import Path
import numpy as np
from transforms import RandomResizedCrop2D, InPainting, OutPainting, Painting, LocalPixelShuffling, RandomWindow, CompressOutOfWindow, RandomGamma, RandomHorizontalFlip, Normalize, Compose
from torchvision import transforms
import gc


class AutoEncodeDataset(Dataset):
    def __init__(self, dataset, general_transform, input_transform, target_transform):
        self.data = dataset
        self.general_transform = general_transform
        self.input_transform = input_transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        image = self.general_transform(self.data[idx])
        return self.input_transform(image), self.target_transform(image)
    
    def __len__(self):
        return len(self.data)


def scan_for_files(root, recursive=False, ext=''):
    if len(ext) > 0 and not ext.startswith('.'):
        ext = ('.'+ext).lower()
    if recursive:
        filelist = []
        for curr_root, _, files in os.walk(root):
            for filename in files:
                if filename.lower().endswith(ext):
                    fullpath = os.path.join(curr_root, filename)
                    if fullpath not in filelist:
                        filelist.append(fullpath)
                        print('File %d: %s' % (len(filelist), fullpath), end="\r")
                    gc.collect()
                    #print(filelist[-1])
        print('\nFound %d files' % (len(filelist)))
        return filelist
    else:
        filelist = [os.path.join(root,f) for f in os.listdir(root) if (os.path.isfile(f) and f.lower().endswith(ext))]
        print('Found %d files' % (len(filelist)))
        return filelist


def apply_window(img, window_center, window_width, clip=False):
    new_wc = np.array(window_center)
    if new_wc.ndim == 0:
        new_wc = new_wc.reshape((1))
    new_ww = np.array(window_width)
    if new_ww.ndim == 0:
        new_ww = new_ww.reshape((1))
    new_wc = new_wc.reshape((new_wc.shape[0],)+(1,)*(img.ndim-new_wc.ndim))
    new_ww = new_ww.reshape((new_ww.shape[0],)+(1,)*(img.ndim-new_ww.ndim))
    
    new_img = ((img - (new_wc - 0.5*new_ww)) / new_ww)
    if clip:
        new_img = new_img.clip(0,1)
    
    return new_img


class PatchDataset2D(Dataset):
    def __init__(self, dataset, h_frac, w_frac, overlap_frac):
        self.dataset = dataset
        self.entries = []
        assert(h_frac <= 1)
        assert(w_frac <= 1)
        for i, image in enumerate(dataset):
            patch_h = int(image.shape[-2] * h_frac)
            patch_w = int(image.shape[-1] * w_frac)
            step_h = int(patch_h * (1 - overlap_frac))
            step_w = int(patch_w * (1 - overlap_frac))
            print('Slicing %d/%d, %dx%d, Patch Size: %dx%d, Step: %d,%d' % (i+1, len(dataset), image.shape[-2], image.shape[-1], patch_h, patch_w, step_h, step_w))
            for y in range(0, image.shape[-2] - patch_h, step_h):
                for x in range(0, image.shape[-1] - patch_w, step_w):
                    #print('patch_h:', patch_h, 'step_h:', step_h, ', patch_w:', patch_w, 'step_w', step_w, ', x:', x, '/', image.shape[-1], ', y:', y, '/', image.shape[-2])
                    self.entries.append({   'index': i,
                                            'slice': (slice(None), slice(y, y + patch_h), slice(x, x + patch_w))
                                        })
                if image.shape[-1] - patch_w > x + 10:
                    x = image.shape[-1] - patch_w
                    #print('patch_h:', patch_h, 'step_h:', step_h, ', patch_w:', patch_w, 'step_w', step_w, ', x:', x, '/', image.shape[-1], ', y:', y, '/', image.shape[-2])
                    self.entries.append({   'index': i,
                                            'slice': (slice(None), slice(y, y + patch_h), slice(x, x + patch_w))
                                        })
            if image.shape[-2] - patch_h > y + 10:
                y = image.shape[-2] - patch_h
                for x in range(0, image.shape[-1] - patch_w, step_w):
                    #print('patch_h:', patch_h, 'step_h:', step_h, ', patch_w:', patch_w, 'step_w', step_w, ', x:', x, '/', image.shape[-1], ', y:', y, '/', image.shape[-2])
                    self.entries.append({   'index': i,
                                            'slice': (slice(None), slice(y, y + patch_h), slice(x, x + patch_w))
                                        })
                if image.shape[-1] - patch_w > x + 10:
                    x = image.shape[-1] - patch_w
                    #print('patch_h:', patch_h, 'step_h:', step_h, ', patch_w:', patch_w, 'step_w', step_w, ', x:', x, '/', image.shape[-1], ', y:', y, '/', image.shape[-2])
                    self.entries.append({   'index': i,
                                            'slice': (slice(None), slice(y, y + patch_h), slice(x, x + patch_w))
                                        })
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        return self.dataset[entry['index']][entry['slice']]
    
    def __len__(self):
        return len(self.entries)


class DicomDataset(Dataset):
    def __init__(self, root, recursive=False, ext='dcm', window_mode='none'):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        
        self.root = root
        if not self._load():
            self.imgfiles = scan_for_files(root, recursive=recursive, ext=ext)
            self._validate_dicom_files(remove=True)
            self._calculate_mean_std()
            self._save()
    
    def __getitem__(self, idx):
        return self._acquire_data(self.imgfiles[idx])
    
    def __len__(self):
        return len(self.imgfiles)

    def _validate_dicom_files(self, remove=True):
        items_to_remove = []
        for idx, path in enumerate(self.imgfiles):
            try:
                print('Validating %d: %s' % (idx, path), end="\r")
                dcm = pydicom.dcmread(path)
                wc= dcm.WindowCenter
                ww = dcm.WindowWidth
                pix = dcm.pixel_array
                assert(pix.ndim == 2)
                assert(pix.size > 1400 * 1200)
            except:
                items_to_remove.append(idx)
        if len(items_to_remove):
            print('\nItems cannot be opened by pydicom:', items_to_remove)
            if remove:
                items_to_remove = sorted(items_to_remove, reverse=True)
                for idx in items_to_remove:
                    del self.imgfiles[idx]
    
    def _acquire_data(self, filename, clip=False):
        dcm = pydicom.dcmread(filename)
        window_center = int(dcm.WindowCenter)
        window_width = int(dcm.WindowWidth)
        img = apply_window(dcm.pixel_array[np.newaxis, :, :], window_center, window_width, clip=clip)
        return torch.Tensor(img)
        #return img
    
    def _calculate_mean_std(self):
        fullset = np.empty((1,0))
        batch_size = 128
        sum_ = 0
        std_ = 0

        sizes = []
        size = 0
        for idx, img in enumerate(self.imgfiles):
            img = self._acquire_data(img, clip=True)
            size += img[0].reshape(-1).shape[0]
            if (idx+1) % batch_size == 0:
                sizes.append(size)
                print('Batch', len(sizes)-1, size)
                size = 0
        if len(self) % batch_size != 0:
            sizes.append(size)
            print('Batch', len(sizes)-1, size, '- last batch')
        
        fullset = np.zeros((img.shape[0], sizes[0]))
        cumulated = 0
        for idx, img in enumerate(self.imgfiles):
            img = self._acquire_data(img, clip=True)
            reshaped = img.reshape((img.shape[0],-1))
            assert(reshaped.min() >= 0)
            assert(reshaped.max() <= 1)
            #fullset = np.concatenate((fullset, img.reshape((img.shape[0],-1))), axis=1)
            fullset[:, cumulated:cumulated+reshaped.shape[1]] = reshaped
            cumulated += reshaped.shape[1]
            #print('Batch', idx % batch_size, idx, cumulated, fullset.shape[1])
            #print(idx, fullset.shape)

            if (idx+1) % batch_size == 0:
                assert(cumulated == fullset.shape[1])
                #windowed_img = LiTSDataset.apply_window(fullset, window_center=self.window_center, window_width=self.window_width)
                sum_ += fullset.sum(axis=1) * batch_size / fullset[0].size
                std_ += fullset.std(axis=1) * batch_size
                #fullset = np.empty((1,0))
                print('Batch',  idx // batch_size, ', sum =', sum_, ', std =', std_, ', mean=', fullset.mean())
                fullset = np.zeros((img.shape[0], sizes[(idx+1) // batch_size]))
                cumulated = 0
        
        if len(self) % batch_size != 0:
            assert(cumulated == fullset.shape[1])
            #windowed_img = LiTSDataset.apply_window(fullset, window_center=self.window_center, window_width=self.window_width)
            sum_ += fullset.sum(axis=1) * ((idx+1) % batch_size) / fullset[0].size
            std_ += fullset.std(axis=1) * ((idx+1) % batch_size)
            print('Batch',  idx // batch_size, '(',  (idx+1) % batch_size, ')', ',sum =', sum_, ',std =', std_)
        
        self.mean = sum_ / len(self)
        self.std = std_ / len(self)
        print('mean:', self.mean, ', std:', self.std)
    
    def _save(self):
        import pickle
        
        filename = os.path.join(self.root, 'imgfiles.pickle')
        with open(filename, 'wb') as file:
            pickle.dump(self.imgfiles, file)
        
        filename = os.path.join(self.root, 'mean.pickle')
        with open(filename, 'wb') as file:
            pickle.dump(self.mean, file)
        
        filename = os.path.join(self.root, 'std.pickle')
        with open(filename, 'wb') as file:
            pickle.dump(self.std, file)
    
    def _load(self):
        import pickle
        filename = os.path.join(self.root, 'imgfiles.pickle')
        if os.path.isfile(filename):
            with open(filename, 'rb') as file:
                self.imgfiles = pickle.load(file)
        else:
            return False
        
        filename = os.path.join(self.root, 'mean.pickle')
        if os.path.isfile(filename):
            with open(filename, 'rb') as file:
                self.mean = pickle.load(file)
        else:
            return False
        
        filename = os.path.join(self.root, 'std.pickle')
        if os.path.isfile(filename):
            with open(filename, 'rb') as file:
                self.std = pickle.load(file)
        else:
            return False
        
        print('Loaded %d files' % (len(self)))
        print('mean:', self.mean, ', std:', self.std)
        
        return True


if __name__ == '__main__':
    import sys
    import cv2
    dataset = DicomDataset(sys.argv[1], recursive=False)
    print(dataset[0].clip(0,1).mean(), dataset[0].clip(0,1).std())
    pdataset = PatchDataset2D(dataset, 256/1120, 256/896, 0.5)
    aedataset = AutoEncodeDataset(dataset, Compose([RandomResizedCrop2D(256), RandomHorizontalFlip()]), Compose([LocalPixelShuffling(), Painting(), RandomWindow(), CompressOutOfWindow(), RandomGamma(), Normalize(0.5, 1)]), Compose([CompressOutOfWindow()]))
    for entry in aedataset:
        input, target = entry
        print(input.shape, input.min(), input.max())
        print(target.shape, target.min(), target.max())
        cv2.imwrite('input.png', ((input.numpy()+0.5).clip(0,1)*255).astype(np.uint8)[0])
        cv2.imwrite('target.png', ((target.numpy()).clip(0,1)*255).astype(np.uint8)[0])
