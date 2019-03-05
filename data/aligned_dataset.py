import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_cv2
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
#import png
import itertools
import cv2
import torch
#from . import util, html


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__old(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        cv2_img = cv2.imread(AB_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH);
        #float_img = cv2_img.astype(float)
        #AB = Image.fromarray(float_img)
        #AB = Image.open(AB_path).convert('RGB')

        AB = Image.open(AB_path).convert('RGB');
        # print(img.getextrema());
        nparr = np.array(AB);
        max_val = np.amax(nparr);
        #
        # split AB image into A and B

        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        cv2_img = cv2.imread(AB_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH);
        #cv2_img = cv2.imread(AB_path, cv2.IMREAD_COLOR);

        depth_div_factor = 65535.0 if cv2_img.dtype == np.uint16 else 255.0
        cv2_img = cv2_img.astype('float32')
        cv2_img = cv2_img[..., ::-1] / depth_div_factor
        # rgb = B_numpy_conv[..., ::-1]
        ht, wd, channels = cv2_img.shape
        half_wd = int(wd/2);

        A_numpy = cv2_img[:, 0:half_wd, :]
        B_numpy = cv2_img[:, half_wd:wd, :]

        # A_numpy = np.copy(A_numpy[:, ::-1, :])
        # B_numpy = np.copy(B_numpy[:, ::-1, :])

        B_r = B_numpy[:,:,2]
        B_g = B_numpy[:,:,1]
        B_b = B_numpy[:,:,0]

        # print("Opening images as 16bit : ", self.opt.is_16_bit);
        # print("Max val of A, B : ", np.amax(A_numpy), np.amax(B_numpy));

        #float_img = cv2_img.astype(float)
        #AB = Image.fromarray(float_img)
        # AB = Image.open(AB_path).convert('RGB')

        # img = Image.open(AB_path).convert('RGB');
        # print(img.getextrema());
        # nparr = np.array(AB);
        # max_val = np.amax(nparr);
        #
        # split AB image into A and B

        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A_transform = get_transform_cv2(self.opt)
        B_transform = get_transform_cv2(self.opt)
        # A_t = torch.from_numpy(A).double()

        A_torch = torch.from_numpy(A_numpy.transpose((2, 0, 1)))
        B_torch = torch.from_numpy(B_numpy.transpose((2, 0, 1)))

        A = A_transform(A_torch)
        B = B_transform(B_torch)

        # A_numpy_conv = A.data.cpu().float().numpy()  # convert it into a numpy array
        # B_numpy_conv = B.data.cpu().float().numpy()  # convert it into a numpy array
        #
        # A_numpy_conv = (((A_numpy_conv + 1) * .5) * 65535.0).astype(np.uint16)
        # B_numpy_conv = (((B_numpy_conv + 1) * .5) * 65535.0).astype(np.uint16)
        #
        # A_numpy_conv = A_numpy_conv.transpose((1, 2, 0))
        # B_numpy_conv = B_numpy_conv.transpose((1, 2, 0))
        #
        #

        # dirs = AB_path.rsplit('\\', 1);
        # dir = dirs[0]
        # full_name = dirs[1]

        # parts = full_name.rsplit('.', 1);
        # name = parts[0]
        # ext = parts[1]

        # cv2.imwrite(dir + '/preview/' + name + '_B_r.' + ext, B_r.astype(np.uint16))
        # cv2.imwrite(dir + '/preview/' + name + '_B_g.' + ext, B_g.astype(np.uint16))
        # cv2.imwrite(dir + '/preview/' + name + '_B_b.' + ext, B_b.astype(np.uint16))

        # cv2.imwrite(name + '_A.' + ext, A_numpy_conv)
        # cv2.imwrite(name + '_B.' + ext, B_numpy_conv)
        #
        # cv2.imwrite(name + '_B-init.' + ext, B_numpy.astype(np.uint16))
        #
        # rgb = B_numpy_conv[..., ::-1]
        # Image.fromarray((rgb/ 255.0).astype(np.uint8)).save(name + '_B-pil.' + ext);

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
