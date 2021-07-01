import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import random
import numpy as np


class RandomWindow(torch.nn.Module):
    def __init__(self, random_state=None):
        super(RandomWindow, self).__init__()
        self.random_state = random_state if random_state is not None else random

    def __call__(self, image):
        wc_factor = self.random_state.uniform(0.9, 1.11)
        if self.random_state.random() < 0.5:
            wc_factor = 1 / wc_factor
        new_wc = 0.5 * wc_factor
        
        ww_factor = self.random_state.uniform(0.9, 1.11)
        if self.random_state.random() < 0.5:
            ww_factor = 1 / ww_factor
        new_ww = 1 * ww_factor
        
        #print(new_wc, new_ww)
        return ((image - (new_wc - 0.5*new_ww)) / new_ww)


class CompressOutOfWindow(torch.nn.Module):
    def __init__(self, mode='clip', random_state=None):
        super(CompressOutOfWindow, self).__init__()
        self.mode = mode
        self.random_state = random_state if random_state is not None else random

    def __call__(self, image):
        new_img = image
            
        if self.mode == 'power':
            new_img[new_img > 1] = np.power(new_img[new_img > 1], 0.1)
            new_img[new_img < 1] = 1 - np.power(1+abs(new_img[new_img < 1]), 0.1)
        elif self.mode == 'power_scale':
            new_img[new_img > 1] = np.power(new_img[new_img > 1], 0.1)
            new_img[new_img < 1] = 1 - np.power(1+abs(new_img[new_img < 1]), 0.1)
            new_img = new_img / (new_img.max(axis=(1,2,3), keepdims=True) - new_img.min(axis=(1,2,3), keepdims=True))
        else:
            new_img = new_img.clip(0, 1)
        
        return new_img


class RandomGamma(torch.nn.Module):
    def __init__(self, random_state=None):
        super(RandomGamma, self).__init__()
        self.random_state = random_state if random_state is not None else random

    def __call__(self, image):
        gamma = self.random_state.uniform(0.66, 1.5)
        if self.random_state.random() < 0.5:
            gamma = 1 / gamma
        
        return image ** gamma


class LocalPixelShuffling(torch.nn.Module):
    def __init__(self, prob=0.5, max_block_size=16, loop=1000, random_state=None):
        super(LocalPixelShuffling, self).__init__()
        self.prob = prob
        self.max_block_size = max_block_size
        self.loop = loop
        self.random_state = random_state if random_state is not None else random
    
    def __call__(self, image):
        if self.random_state.random() >= self.prob:
            return image
        
        if isinstance(image, torch.Tensor):
            image_temp = image.clone()
            orig_image = image.clone()
        else:
            image_temp = image.copy()
            orig_image = image.copy()

        for _ in range(self.loop):
            window_sizes = []
            #window_positions = []
            slices = []
            for d in range(image.ndim-1):
                window_size = (self.random_state.randint(1, min(self.max_block_size, image.shape[d+1])))
                window_position = (self.random_state.randint(0, image.shape[d+1] - window_size))
                #print(window_position, window_position + window_size)
                window_sizes.append(window_size)
                slices.append(slice(window_position, window_position + window_size))
                
            #print(slices, window_sizes)

            for c in range(image.shape[0]):
                window = orig_image[tuple([c]+slices)]
                if isinstance(window, torch.Tensor):
                    idx = torch.randperm(window.nelement())
                    window = window.reshape(-1)[idx].reshape(window.size())
                else:
                    window = window.flatten()
                    self.random_state.shuffle(window)
                    window = window.reshape(tuple(window_sizes))
                image_temp[tuple([c]+slices)] = window

        return image_temp


class InPainting(torch.nn.Module):
    def __init__(self, prob=0.95, random_state=None):
        super(InPainting, self).__init__()
        self.prob = prob
        self.random_state = random_state if random_state is not None else random
    
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            new_img = image.clone()
        else:
            new_img = image.copy()

        count = 5
        while count > 0 and self.random_state.random() < 0.95:
            window_sizes = [image.shape[0]]
            #window_positions = []
            slices = [slice(None)]
            for d in range(image.ndim-1):
                window_size = (self.random_state.randint(image.shape[d+1]//8, image.shape[d+1]//4))
                window_position = (self.random_state.randint(3, image.shape[d+1] - window_size))
                #print(window_position, window_position + window_size)
                window_sizes.append(window_size)
                slices.append(slice(window_position, window_position + window_size))
                
            #print(slices, window_sizes)
            if isinstance(new_img, torch.Tensor):
                new_img[slices] = torch.rand(*window_sizes)
            else:
                new_img[tuple(slices)] = np.random.rand(*window_sizes)
            
            count -= 1
        
        return new_img


class OutPainting(torch.nn.Module):
    def __init__(self, prob=0.95, random_state=None):
        super(OutPainting, self).__init__()
        self.prob = prob
        self.random_state = random_state if random_state is not None else random
    
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            new_img = torch.rand(image.shape)
        else:
            if isinstance(self.random_state, np.random.RandomState):
                new_img = self.random_state.rand(*image.shape)
            else:
                new_img = np.random.rand(*image.shape)

        count = 6
        while count > 0 and self.random_state.random() < 0.95:
            window_sizes = [image.shape[0]]
            #window_positions = []
            slices = [slice(None)]
            for d in range(image.ndim-1):
                window_size = (self.random_state.randint(3*image.shape[d+1]//7, 4*image.shape[d+1]//7))
                window_position = (self.random_state.randint(3, image.shape[d+1] - window_size - 3))
                #print(window_position, window_position + window_size)
                window_sizes.append(window_size)
                slices.append(slice(window_position, window_position + window_size))
                
            #print(slices, window_sizes)
            new_img[tuple(slices)] = image[tuple(slices)]
            count -= 1
        
        return new_img


class Painting(torch.nn.Module):
    def __init__(self, inpainting_prob=0.5, random_state=None):
        super(Painting, self).__init__()
        self.inpainting_prob = inpainting_prob
        self.inpainting = InPainting(random_state=random_state)
        self.outpainting = OutPainting(random_state=random_state)
        self.random_state = random_state if random_state is not None else random
    
    def __call__(self, image):
        if self.random_state.random() < self.inpainting_prob:
            return self.inpainting(image)
        else:
            return self.outpainting(image)


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5, random_state=None):
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob
        self.random_state = random_state if random_state is not None else random
    
    def __call__(self, image):
        if self.random_state.random() < self.prob:
            return image
        else:
            if isinstance(image, torch.Tensor):
                return image.flip(-1)
            else:
                return np.flip(image, axis=-1)


class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            return F.normalize(image, self.mean, self.std, self.inplace)
        else:
            return (image - self.mean) / self.std


class RandomResizedCrop2D(torchvision.transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.055, 0.075), ratio=(0.8, 1.25), interpolation=InterpolationMode.BILINEAR):
        super(RandomResizedCrop2D, self).__init__(size, scale, ratio, interpolation)
    
    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            image = torch.Tensor(image)
        cropped = super(RandomResizedCrop2D, self).__call__(image)
        count = 0
        while cropped.mean() < 0.01:
            count += 1
            cropped = super(RandomResizedCrop2D, self).__call__(image)
            if count > 20:
                break
        return cropped


class Resize2D(torchvision.transforms.Resize):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super(Resize2D, self).__init__(size, interpolation, max_size, antialias)
    
    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            image = torch.Tensor(image)
        
        return super(Resize2D, self).__call__(image)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


if __name__ == '__main__':
    testt = Compose([RandomResizedCrop2D(64), Painting(), LocalPixelShuffling(), RandomWindow(), CompressOutOfWindow(), RandomGamma(), RandomHorizontalFlip(), Normalize(0.5, 1)])
    #t = torch.Tensor([[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    t, _ = torch.sort(torch.rand((1,128,128)))
    tt = testt(t)
    print(t.shape)
    print(tt.shape)

    #t = np.array([[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    #t = np.sort(np.random.rand(1,32,32))
    ta = t.numpy()
    tt = testt(ta)
    print(ta.shape)
    print(tt.shape)
