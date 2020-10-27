import torch
import numpy as np
import parameters as pt
from torch.utils.data import Dataset
from numpy.random import normal, randint, permutation
from glob import glob
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import prewitt
from os import makedirs
from os.path import dirname, exists
from scipy.ndimage.filters import convolve
from scipy import fftpack


def _pad_to(image, size):
    Hk, Wk = image.shape
    Ht, Wt = size
    Hp, Wp = (Ht - Hk)//2, (Wt - Wk)//2

    return np.pad(image, pad_width=((Hp, Ht - Hk - Hp), (Wp, Wt - Wk - Wp)),
                  mode='constant')


def _ensure_gray(image):
    if image.ndim <= 2:
        return image
    elif image.shape[2] == 1:
        return image[:, :, 0]
    else:
        return rgb2gray(image)


def _augment(image):
    transformed = np.fliplr(image) if randint(2) == 1 else image
    transformed = np.rot90(transformed, k=randint(4))

    return transformed


def list_image_files(image_dir):
    '''
    List image files under "image_dir"
    '''

    image_suffices = ['jpg', 'png', 'bmp', 'gif']
    image_files = []
    for suffix in image_suffices:
        image_files += sorted(glob(image_dir + '/*.' + suffix))

    return image_files


def load_image(image_file):
    '''
    Read image, normalize and ensure even size
    '''

    image = imread(image_file)
    image = img_as_float(image)
    if image.ndim == 2:  # grayscale
        image = np.expand_dims(image, axis=-1)
    elif image.shape[2] == 3 and pt.image_channels == 1:
        image = np.expand_dims(rgb2gray(image), axis=-1)
    Hi, Wi, _ = image.shape
    image = image[:2*(Hi//2), :2*(Wi//2), :]

    return image.astype('float32')


def imwrite(image, file_path):
    root = dirname(file_path)
    if not exists(root):
        makedirs(root)
    # Clip image to [0, 1]
    image[image < 0] = 0
    image[image > 1] = 1
    imsave(file_path, image)


def load_kernel(kernel_file, bounding_box_size=None):
    '''
    Read kernel, normalize and ensure odd size
    '''

    kernel = imread(kernel_file, as_gray=True).astype('float32')
    kernel /= kernel.sum()
    if bounding_box_size:
        return _pad_to(kernel, bounding_box_size)
    else:
        Hk, Wk = kernel.shape
        return np.pad(kernel, pad_width=((0, (Hk + 1) % 2), (0, (Wk + 1) % 2)),
                      mode='constant')


def to_tensor(array):
    '''
    Convert numpy array to pytorch tensor
    '''

    if array.ndim == 3:  # HxWxC
        array = np.transpose(array, axes=(2, 0, 1))
    elif array.ndim == 4:  # NxHxWxC
        array = np.transpose(array, axes=(0, 3, 1, 2))

    return torch.from_numpy(array.astype('float32'))


def parse(data, device=torch.device('cpu')):
    blur = data['blurred'].to(device)
    image = data['image'].to(device)
    kernel = data['kernel'].to(device)

    return blur, image, kernel


def random_crop(image, patch_size):
    Hi, Wi, _ = image.shape
    Hp, Wp = patch_size
    h0 = 0 if Hi == Hp else randint(0, Hi - Hp)
    w0 = 0 if Wi == Wp else randint(0, Wi - Wp)

    return image[h0: h0 + Hp, w0: w0 + Wp, :]


def convn(image, kernel):
    '''
    Multi-dimensional convolution with 'valid' padding
    '''

    Hk, Wk = kernel.shape
    Hk2, Wk2 = Hk // 2, Wk // 2

    def conv2(x, k):
        return convolve(x, k, mode='constant')[Hk2:-Hk2, Wk2:-Wk2]
    if image.ndim < 3:
        return conv2(image, kernel)

    num_channels = image.shape[2]
    if num_channels == 1:
        return np.expand_dims(conv2(image[:, :, 0], kernel), axis=-1)
    else:
        channels = [conv2(image[:, :, c], kernel) for c in range(num_channels)]
        return np.stack(channels, axis=-1)


def solve_min_laplacian(boundary_image):
    H, W = boundary_image.shape

    # Laplacian
    f = np.zeros((H, W))

    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0
    j = np.arange(1, H - 1)
    k = np.arange(1, W - 1)
    f_bp = np.zeros((H, W))
    f_bp[np.ix_(j, k)] = -4*boundary_image[np.ix_(j, k)] \
        + boundary_image[np.ix_(j, k + 1)] + boundary_image[np.ix_(j, k - 1)] \
        + boundary_image[np.ix_(j - 1, k)] + boundary_image[np.ix_(j + 1, k)]

    # subtract boundary points contribution
    f1 = f - f_bp  # subtract boundary points contribution

    # DST Sine Transform algo starts here
    f2 = f1[1:-1, 1:-1]
    # compute sine tranform

    def dst(x):
        return fftpack.dst(x, type=1, axis=0) / 2.0

    def idst(x):
        return np.real(fftpack.idst(x, type=1, axis=0)) / (x.shape[0] + 1.0)

    tt = dst(f2)
    f2sin = dst(tt.T).T

    # compute Eigen Values
    x, y = np.meshgrid(np.arange(1, W - 1), np.arange(1, H - 1))
    denom = 2*np.cos(np.pi*x/(W-1)) - 2 + 2*np.cos(np.pi*y/(H-1)) - 2

    # divide
    f3 = f2sin / denom

    # compute Inverse Sine Transform
    tt = idst(f3)
    img_tt = idst(tt.T).T

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image
    img_direct[1:-1, 1:-1] = 0
    img_direct[1:-1, 1:-1] = img_tt

    return img_direct


def edgetaper(blurred, kernel_size):
    Hk, Wk = kernel_size
    Hk2, Wk2 = Hk // 2, Wk // 2
    padded = np.pad(blurred, pad_width=((Hk2, Hk - Hk2 - 1),
                                        (Wk2, Wk - Wk2 - 1), (0, 0)),
                    mode='linear_ramp')
    # Pad in four directions
    for c in range(padded.shape[2]):
        padded[:Hk2+1, Wk2:-Wk2, c] = solve_min_laplacian(
            padded[:Hk2+1, Wk2:-Wk2, c])
        padded[-Hk2-1:, Wk2:-Wk2, c] = solve_min_laplacian(
            padded[-Hk2-1:, Wk2:-Wk2, c])
        padded[:, :Wk2+1, c] = solve_min_laplacian(padded[:, :Wk2+1, c])
        padded[:, -Wk2-1:, c] = solve_min_laplacian(padded[:, -Wk2-1:, c])
    padded = np.pad(padded, pad_width=((Hk//2, Hk - Hk//2 - 1),
                                       (Wk//2, Wk - Wk//2 - 1), (0, 0)),
                    mode='constant')

    return padded


class SyntheticDataset(Dataset):
    def __init__(self, image_dir, kernel_dir,
                 max_trial=10, grad_thr=0.05, thr_ratio=0.06):
        self.image_files = list_image_files(image_dir)
        self.kernel_files = list_image_files(kernel_dir)
        self.kernel_indices = permutation(len(self.kernel_files))
        self.max_trial = max_trial
        self.grad_thr = grad_thr
        self.thr_ratio = thr_ratio

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        Hv, Wv = pt.patch_size  # 'valid' size
        Hk, Wk = pt.bounding_box_size
        Hp, Wp = Hv + Hk - 1, Wv + Wk - 1  # 'same' size
        image = _augment(load_image(self.image_files[idx]))
        # Hi, Wi = image.shape[0], image.shape[1]
        # Hv, Wv = Hi - Hk + 1, Wi - Wk + 1
        # patch = resize(image, (Hp, Wp), mode='reflect', anti_aliasing=True)
        for t in range(self.max_trial):
            patch = random_crop(image, (Hp, Wp))
            # Validate patch: reject it if it is over-smooth
            grad = prewitt(_ensure_gray(patch))
            ratio = np.count_nonzero(grad > self.grad_thr) / float(grad.size)
            if ratio > self.thr_ratio:
                break
        ker_idx = self.kernel_indices[idx]
        kernel = load_kernel(self.kernel_files[ker_idx], pt.bounding_box_size)
        blurred = convn(patch, kernel)
        # blurred += normal(scale=pt.noise_stddev, size=blurred.shape)
        # Pad the invisible boundary region
        # blurred = to_tensor(np.pad(blurred, pad_width=((Hk - 1, Hk - 1),
                                                       # (Wk - 1, Wk - 1),
                                                       # (0, 0)),
                                   # mode='constant'))
        # blurred = to_tensor(edgetaper(blurred, (Hk, Wk)))
        blurred = to_tensor(blurred)
        patch = to_tensor(patch[Hk//2:Hk//2 + Hv, Wk//2:Wk//2 + Wv, :])

        return {'blurred': blurred, 'image': patch,
                'kernel': to_tensor(kernel)}


class BlurredImageDataset(Dataset):
    def __init__(self, data_dir):
        self.blur_image_files = list_image_files(data_dir + '/blurred')
        self.sharp_image_files = list_image_files(data_dir + '/sharp')
        assert len(self.blur_image_files) == len(self.sharp_image_files)
        self.kernel_files = list_image_files(data_dir + '/kernel')
        self.num_kernels = len(self.kernel_files)
        self.num_images = len(self.blur_image_files)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        Hk, Wk = pt.bounding_box_size
        blurred = load_image(self.blur_image_files[idx])
        image = load_image(self.sharp_image_files[idx])
        # Hi, Wi, _ = image.shape
        # Hp, Wp = pt.patch_size
        # h0 = 0 if Hi == Hp else randint(0, Hi - Hp)
        # w0 = 0 if Wi == Wp else randint(0, Wi - Wp)
        # blurred = blurred[h0: h0 + Hp, w0: w0 + Wp, :]
        # image = image[h0: h0 + Hp, w0: w0 + Wp, :]
        # blurred += normal(scale=pt.noise_stddev, size=blurred.shape)
        # blurred = to_tensor(edgetaper(blurred, (Hk, Wk)))
        blurred = to_tensor(blurred)
        image = to_tensor(image)
        kernel = None if self.num_kernels == 0 else to_tensor(
            load_kernel(self.kernel_files[idx], pt.bounding_box_size))

        return {'blurred': blurred, 'image': image, 'kernel': kernel}
