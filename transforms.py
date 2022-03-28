import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import random
import numpy as np
import cv2


class Invert(torch.nn.Module):
    def __init__(self, p=0.3, random_state=None):
        super(Invert, self).__init__()
        self.p = p
        self.random_state = random_state if random_state is not None else random

    def __call__(self, image):
        if self.random_state.random() > self.p:
            return image
        else:
            return 1-image


class HalfResolution(torch.nn.Module):
    def __init__(self, p=0.1, random_state=None):
        super(HalfResolution, self).__init__()
        self.p = p
        self.random_state = random_state if random_state is not None else random

    def __call__(self, image):
        if self.random_state.random() > self.p:
            return image

        if isinstance(image, torch.Tensor):
            #interpolation = self.random_state.choice(list(InterpolationMode))
            interpolation = self.random_state.choice([InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC])
            out = F.resize(image, [image.shape[-2]//2, image.shape[-1]//2], interpolation)
            out = F.resize(out, [image.shape[-2], image.shape[-1]], InterpolationMode.NEAREST)
        else:
            interpolation = self.random_state.choice([cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
            out = cv2.resize(image[0] if image.ndim == 3 else image, [image.shape[-1]//2, image.shape[-2]//2], interpolation = interpolation)
            out = cv2.resize(out, [image.shape[-1], image.shape[-2]], interpolation = interpolation)
        return out


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
        gamma = self.random_state.uniform(2./3, 3./2)
        if self.random_state.random() < 0.5:
            gamma = 1 / gamma
        
        return image ** gamma


class LocalPixelShuffling(torch.nn.Module):
    def __init__(self, prob=0.5, max_block_size=8, loop=1000, random_state=None):
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
                    if isinstance(self.random_state, np.random.RandomState):
                        window_np = window.numpy()
                        window_np = window_np.flatten()
                        self.random_state.shuffle(window_np)
                        window_np = window_np.reshape(tuple(window_sizes))
                        window = torch.Tensor(window_np)
                    else:
                        idx = torch.randperm(window.nelement())
                        window = window.reshape(-1)[idx].reshape(window.size())
                else:
                    window = window.flatten()
                    self.random_state.shuffle(window)
                    window = window.reshape(tuple(window_sizes))
                image_temp[tuple([c]+slices)] = window

        return image_temp


class InPainting(torch.nn.Module):
    def __init__(self, prob=0.95, fill_mode='noise', max_blocksize=32, random_state=None):
        super(InPainting, self).__init__()
        self.prob = prob
        self.fill_mode = fill_mode
        self.max_blocksize = max_blocksize
        self.random_state = random_state if random_state is not None else random
    
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            new_img = image.clone()
        else:
            new_img = image.copy()

        if self.fill_mode == 'random':
            fill_mode = self.random_state.choice(['noise', 'average', 'zero'])
        else:
            fill_mode = self.fill_mode
        
        count = 16
        while count > 0 and self.random_state.random() < 0.95:
            window_sizes = [image.shape[0]]
            #window_positions = []
            slices = [slice(None)]
            for d in range(image.ndim-1):
                window_size = (self.random_state.randint(self.max_blocksize/2, self.max_blocksize))
                window_position = (self.random_state.randint(3, image.shape[d+1] - window_size))
                #print(window_position, window_position + window_size)
                window_sizes.append(window_size)
                slices.append(slice(window_position, window_position + window_size))
                
            #print(slices, window_sizes)
            if fill_mode == 'noise':
                if isinstance(new_img, torch.Tensor):
                    if isinstance(self.random_state, np.random.RandomState):
                        new_img[slices] = torch.Tensor(self.random_state.rand(*window_sizes))#torch.rand(*window_sizes)
                    else:
                        new_img[slices] = torch.rand(*window_sizes)
                else:
                    if isinstance(self.random_state, np.random.RandomState):
                        new_img[tuple(slices)] = self.random_state.rand(*window_sizes)
                    else:
                        new_img[tuple(slices)] = np.random.rand(*window_sizes)
            if fill_mode == 'zero':
                if isinstance(new_img, torch.Tensor):
                    new_img[slices] = 0.0
                else:
                    if isinstance(self.random_state, np.random.RandomState):
                        new_img[tuple(slices)] = 0.0
                    else:
                        new_img[tuple(slices)] = 0.0
            elif fill_mode == 'average':
                new_img[slices] = new_img[slices].mean()
            
            count -= 1
        
        return new_img


class OutPainting(torch.nn.Module):
    def __init__(self, prob=0.95, fill_mode='noise', random_state=None):
        super(OutPainting, self).__init__()
        self.prob = prob
        self.fill_mode = fill_mode
        self.random_state = random_state if random_state is not None else random
    
    def __call__(self, image):
        if self.fill_mode == 'random':
            fill_mode = self.random_state.choice(['noise', 'average', 'zero'])
        else:
            fill_mode = self.fill_mode
        
        if fill_mode == 'noise':
            if isinstance(image, torch.Tensor):
                if isinstance(self.random_state, np.random.RandomState):
                    new_img = torch.Tensor(self.random_state.rand(*image.shape))#torch.rand(image.shape)
                else:
                    new_img = torch.rand(image.shape)
            else:
                if isinstance(self.random_state, np.random.RandomState):
                    new_img = self.random_state.rand(*image.shape, dtype=np.float32)
                else:
                    new_img = np.random.rand(*image.shape, dtype=np.float32)
        elif fill_mode == 'average':
            if isinstance(image, torch.Tensor):
                new_img = torch.full(image.shape, image.mean(), dtype=torch.float32)
            else:
                new_img = np.full(image.shape, image.mean(), dtype=np.float32)
        elif fill_mode == 'zero':
            if isinstance(image, torch.Tensor):
                new_img = torch.zeros(image.shape, dtype=torch.float32)
            else:
                new_img = np.zeros(image.shape, dtype=np.float32)
        else:
            raise NotImplementedError

        count = 8
        while count > 0:# and self.random_state.random() < 0.95:
            window_sizes = [image.shape[0]]
            #window_positions = []
            slices = [slice(None)]
            for d in range(image.ndim-1):
                window_size = (self.random_state.randint(2*image.shape[d+1]//4, 3*image.shape[d+1]//4))
                window_position = (self.random_state.randint(3, image.shape[d+1] - window_size - 3))
                #print(window_position, window_position + window_size)
                window_sizes.append(window_size)
                slices.append(slice(window_position, window_position + window_size))
                
            #print(slices, window_sizes)
            new_img[tuple(slices)] = image[tuple(slices)]
            count -= 1
        
        return new_img


class Painting(torch.nn.Module):
    def __init__(self, inpainting_prob=0.8, fill_mode='noise', random_state=None):
        super(Painting, self).__init__()
        self.inpainting_prob = inpainting_prob
        self.inpainting = InPainting(fill_mode=fill_mode, random_state=random_state)
        self.outpainting = OutPainting(fill_mode=fill_mode, random_state=random_state)
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
    random_state = np.random.RandomState(seed=0)
    testt = Compose([Resize2D(64), LocalPixelShuffling(prob=1.0, random_state=random_state), Painting(random_state=random_state), Normalize(0.5, 1)])
    
    t, _ = torch.sort(torch.rand((1,128,128)))
    tt = testt(t)
    print(t.shape)
    print(tt.shape)

    testt.transforms[1].random_state.seed(0)
    ta = t.numpy()
    tta = testt(ta)
    print(ta.shape)
    print(tta.shape)

    np.testing.assert_allclose(tt.numpy(), tta)
