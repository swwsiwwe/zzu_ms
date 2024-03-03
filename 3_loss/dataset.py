import cv2
import torch
from scipy.ndimage.morphology import distance_transform_edt
from config import Config
import os
from skimage import io, transform, color
import numpy as np
from PIL import Image
import random

class RandomFlip(object):

    def __init__(self, p = 0.5):
        self.p = p
    def __call__(self, sample):

        image,seg_mask,seg_loss_mask = sample
        horiz = random.random()
        vertical = random.random()
        
        if horiz > self.p : #水平翻转
            image = np.flip(image,axis = 1).copy()
            seg_mask = np.flip(seg_mask,axis = 1).copy()
            seg_loss_mask = np.flip(seg_loss_mask,axis = 1).copy()

        if vertical > self.p : #垂直翻转
            image = np.flip(image,axis = 0).copy()
            seg_mask = np.flip(seg_mask,axis = 0).copy()
            seg_loss_mask = np.flip(seg_loss_mask,axis = 0).copy()
        
        return image,seg_mask,seg_loss_mask

class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, label= sample

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)
        
        
        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True)
        return img,lbl

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):


        image, label = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]
        return image,label



class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

            image, label = sample
            label = label[:, :, np.newaxis]
            # if (np.max(label) < 1e-6):
            #     label = label
            # else:
            #     label = label / np.max(label)
            # change the color space
            if self.flag == 2: # with rgb and Lab colors
                        tmpImg = np.zeros((image.shape[0],image.shape[1],6))
                        tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
                        if image.shape[2]==1:
                            tmpImgt[:,:,0] = image[:,:,0]
                            tmpImgt[:,:,1] = image[:,:,0]
                            tmpImgt[:,:,2] = image[:,:,0]
                        else:
                            tmpImgt = image
                        tmpImgtl = color.rgb2lab(tmpImgt)

                        # nomalize image to range [0,1]
                        tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
                        tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
                        tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
                        tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
                        tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
                        tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

                        # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

                        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
                        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
                        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
                        tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
                        tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
                        tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

            elif self.flag == 1: #with Lab color
                        tmpImg = np.zeros((image.shape[0],image.shape[1],3))

                        if image.shape[2]==1:
                            tmpImg[:,:,0] = image[:,:,0]
                            tmpImg[:,:,1] = image[:,:,0]
                            tmpImg[:,:,2] = image[:,:,0]

                        tmpImg = color.rgb2lab(tmpImg)

                        # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

                        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
                        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
                        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

                        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
                        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
                        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

            else: # with rgb color
                        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
                        image = image/np.max(image)
                        if image.shape[2]==1:
                            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
                            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
                        else:
                            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
                            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225



            tmpImg = tmpImg.transpose((2, 0, 1))
            label = label.transpose((2, 0, 1))

            return torch.from_numpy(tmpImg), torch.from_numpy(label)


class SalObjDataset():
    def __init__(self, image_root, gt_root,transform):
        self.images = [image_root + f   for f in os.listdir(image_root) ]
        self.gts = [gt_root + f  for f in os.listdir(gt_root)]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.transform = transform

    def __getitem__(self, index):
        label = Image.open(self.gts[index])
        img =Image.open(self.images[index]).convert('RGB')
        img=np.array(img)
        label=np.array(label)
        sample = [img,label]
        image,label = self.transform(sample)
        return image,label

    def __len__(self):
        return len(self.images)
class SalObjDataset_order(SalObjDataset):
    def __init__(self, image_root, gt_root,transform):
        self.images = [image_root + f   for f in os.listdir(image_root) ]
        self.gts = [gt_root + f  for f in os.listdir(gt_root)]

        files_image = os.listdir(image_root)
        files_image.sort(key=lambda x: int(x.split('_')[0]))
        files_gt = os.listdir(image_root)
        files_gt.sort(key=lambda x: int(x.split('_')[0]))

        self.images = [os.path.join(image_root, name) for name in files_image]
        self.gts = [os.path.join(gt_root, name) for name in files_gt]

        self.transform = transform

class Dataset():
    def __init__(self, path: str, cfg: Config, kind: str):
        super(Dataset, self).__init__()
        self.path: str = path
        self.cfg: Config = cfg
        self.kind: str = kind
        self.image_size: (int, int) = (self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT)
        self.grayscale: bool = self.cfg.INPUT_CHANNELS == 1

        self.num_negatives_per_one_positive: int = 1
        self.frequency_sampling: bool = self.cfg.FREQUENCY_SAMPLING and self.kind == 'TRAIN'

    def init_extra(self):
        self.counter = 0
        self.neg_imgs_permutation = np.random.permutation(self.num_neg)
        self.pos_imgs_permutation = np.random.permutation(self.num_neg)

        self.neg_retrieval_freq = np.zeros(shape=self.num_neg)
        self.pos_retrieval_freq = np.zeros(shape=self.num_pos)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor, bool, str):
        if self.counter >= self.len:
            self.counter = 0
            if self.frequency_sampling:
                sample_probability = 1 - (self.neg_retrieval_freq / np.max(self.neg_retrieval_freq))
                sample_probability = sample_probability - np.median(sample_probability) + 1
                sample_probability = sample_probability ** (np.log(len(sample_probability)) * 4)
                sample_probability = sample_probability / np.sum(sample_probability)

                # use replace=False for to get only unique values
                self.neg_imgs_permutation = np.random.choice(range(self.num_neg),
                                                             size=self.num_negatives_per_one_positive * self.num_pos,
                                                             p=sample_probability,
                                                             replace=False)
            else:
                self.neg_imgs_permutation = np.random.permutation(self.num_neg)

            self.pos_imgs_permutation = np.random.permutation(self.num_pos)

        if self.kind == 'TRAIN':
            if index >= self.num_pos:
                ix = index % self.num_pos
                ix = self.neg_imgs_permutation[ix]
                item = self.neg_samples[ix]
                self.neg_retrieval_freq[ix] = self.neg_retrieval_freq[ix] + 1

            else:
                ix = index
                item = self.pos_samples[ix]
        else:
            if index < self.num_neg:
                ix = index
                item = self.neg_samples[ix]
            else:
                ix = index - self.num_neg
                item = self.pos_samples[ix]

        image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, sample_name = item

        if self.cfg.ON_DEMAND_READ:  # STEEL only so far
            if image_path == -1 or seg_mask_path == -1:
                raise Exception('For ON_DEMAND_READ image and seg_mask paths must be set in read_contents')
            img = self.read_img_resize(image_path, self.grayscale, self.image_size)
            if seg_mask_path is None:  # good sample
                seg_mask = np.zeros_like(img)
            elif isinstance(seg_mask_path, list):
                seg_mask = self.rle_to_mask(seg_mask_path, self.image_size)
            else:
                seg_mask, _ = self.self.read_label_resize(seg_mask_path, self.image_size)

            if np.max(seg_mask) == np.min(seg_mask):  # good sample
                seg_loss_mask = np.ones_like(seg_mask)
            else:
                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)

            image = self.to_tensor(img)
            seg_mask = self.to_tensor(self.downsize(seg_mask))
            seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))

        self.counter = self.counter + 1

        return image, seg_mask, seg_loss_mask, is_segmented, sample_name

    def __len__(self):
        return self.len

    def read_contents(self):
        pass

    def read_img_resize(self, path, grayscale, resize_dim) -> np.ndarray:

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        
        if resize_dim is not None:
            img = cv2.resize(img, dsize=resize_dim)

        return np.array(img, dtype=np.float32) / 255.0

    def read_label_resize(self, path, resize_dim, dilate=None) -> (np.ndarray, bool):
        lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if dilate is not None and dilate > 1:
            lbl = cv2.dilate(lbl, np.ones((dilate, dilate)))
        if resize_dim is not None:
            lbl = cv2.resize(lbl, dsize=resize_dim)
        return np.array((lbl / 255.0), dtype=np.float32), np.max(lbl) > 0

    def to_tensor(self, x) -> torch.Tensor:
        # if np.max(x) > 1.0:
        if x.dtype != np.float32:
            x = (x / 255.0).astype(np.float32)

        if len(x.shape) == 3:
            x = np.transpose(x, axes=(2, 0, 1))
        else:
            x = np.expand_dims(x, axis=0)

        x = torch.from_numpy(x)
        return x

    def distance_transform(self, mask: np.ndarray, max_val: float, p: float) -> np.ndarray:
        dst_trf = distance_transform_edt(mask)

        if dst_trf.max() > 0:
            dst_trf = (dst_trf / dst_trf.max())
            dst_trf = (dst_trf ** p) * max_val
        
        dst_trf[mask == 0] = 1.0
        return np.array(dst_trf, dtype=np.float32)
    def distance_transform_inside(self, mask: np.ndarray, max_val: float, p: float,mask_inside: np.ndarray) -> np.ndarray:
        dst_trf = distance_transform_edt(mask)

        if dst_trf.max() > 0:
            dst_trf = (dst_trf / dst_trf.max())
            dst_trf = (dst_trf ** p) * max_val
        
        dst_trf[mask == 0] = 1.0
        dst_trf[mask_inside > 0] = 1.0
        return np.array(dst_trf, dtype=np.float32)
    
    def downsize(self, image: np.ndarray, downsize_factor: int = 8) -> np.ndarray:
        img_t = torch.from_numpy(np.expand_dims(image, 0 if len(image.shape) == 3 else (0, 1)).astype(np.float32))
        # img_t = torch.from_numpy(np.expand_dims(image, 0).astype(np.float32))
        img_t = torch.nn.ReflectionPad2d(padding=(downsize_factor))(img_t)
        image_np = torch.nn.AvgPool2d(kernel_size=2 * downsize_factor + 1, stride=downsize_factor)(img_t).detach().numpy()
        return image_np[0] if len(image.shape) == 3 else image_np[0, 0]

    def rle_to_mask(self, rle, image_size):
        if len(rle) % 2 != 0:
            raise Exception('Suspicious')

        w, h = image_size
        mask_label = np.zeros(w * h, dtype=np.float32)

        positions = rle[0::2]
        length = rle[1::2]
        for pos, le in zip(positions, length):
            mask_label[pos - 1:pos + le - 1] = 1
        mask = np.reshape(mask_label, (h, w), order='F')
        return mask
