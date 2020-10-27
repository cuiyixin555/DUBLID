import torch
import torch.nn.functional as F


def upscale(low_res, size):
    '''
    Upsampling 3D or 4D tensors on the last two dimensions
    '''

    if low_res.dim() == 3:
        return F.upsample(low_res.unsqueeze(dim=0), size,
                          mode='bilinear', align_corners=False)[0]
    else:
        return F.upsample(low_res, size, mode='bilinear', align_corners=False)


def downscale(high_res):
    '''
    Downsampling 3D or 4D tensors on the last two dimensions
    '''

    if high_res.dim() == 3:
        return F.avg_pool2d(high_res.unsqueeze(dim=0), kernel_size=2)[0]
    else:
        return F.avg_pool2d(high_res, kernel_size=2)


def conv2(tensor, kernel, mode='same', pad_mode='reflect'):
    '''
    Convolution with output size the same as the 'full' convolution scheme
    '''

    Hk, Wk = kernel.shape[-2], kernel.shape[-1]
    if mode == 'same':
        pad_size = (Wk//2, Wk - Wk//2 - 1, Hk//2, Hk - Hk//2 - 1)
    elif mode == 'full':
        pad_size = (Wk - 1, Wk - 1, Hk - 1, Hk - 1)
    else:  # 'valid'
        pad_size = (0, 0, 0, 0)

    return F.conv2d(F.pad(tensor, pad=pad_size, mode=pad_mode), kernel)


def real(c):
    '''
    Extract real part of complex tensor c
    '''

    return c[..., 0]


def real_mul(r, c):
    '''
    Multiply real tensor r with complex tensor c
    '''

    return r.unsqueeze(dim=-1)*c


def mul(c1, c2):
    '''
    Complex multiplication between c1 and c2
    '''

    r1, i1 = c1[..., 0], c1[..., 1]
    r2, i2 = c2[..., 0], c2[..., 1]
    r = r1*r2 - i1*i2
    c = r1*i2 + i1*r2

    return torch.stack([r, c], dim=-1)


def conj_mul(c1, c2):
    '''
    Complex conjugate of c1 and multiplication with c2
    '''

    r1, i1 = c1[..., 0], -c1[..., 1]
    r2, i2 = c2[..., 0], c2[..., 1]
    r = r1*r2 - i1*i2
    c = r1*i2 + i1*r2

    return torch.stack([r, c], dim=-1)


def csquare(c):
    '''
    Square of absolute values of complex numbers
    '''

    return c[..., 0]**2 + c[..., 1]**2


def pad_to(original, size):
    '''
    Post-pad last two dimensions to "size"
    '''

    original_size = original.size()
    pad = [0, size[1] - original_size[-1],
           0, size[0] - original_size[-2]]

    return F.pad(original, pad)


def fft2(signal, size=None):
    '''
    Fast Fourier transform on the last two dimensions
    '''

    padded = signal if size is None else pad_to(signal, size)

    return torch.rfft(padded, signal_ndim=2)


def ifft2(signal, size=None):
    '''
    Inverse fast Fourier transform on the last two dimensions
    '''

    return torch.irfft(signal, signal_ndim=2, signal_sizes=size)


def circ_shift(ts, shift):
    '''
    Circular shift on the last two dimensions
    '''

    sr, sc = shift
    if sc != 0:  # column shift
        ts = torch.cat((ts[..., sc:], ts[..., :sc]), dim=-1)
    if sr != 0:  # row shift
        ts = torch.cat((ts[..., sr:, :], ts[..., :sr, :]), dim=-2)

    return ts


def image_shift(im, shift):
    '''
    Shift on the last two dimensions
    '''

    sr, sc = shift[0].item(), shift[1].item()
    dim = im.ndimension()
    if dim == 2:
        im = im.unsqueeze(0).unsqueeze(0)
    elif dim == 3:
        im = im.unsqueeze(0)
    if sr > 0:
        im = F.pad(im[..., :-sr, :], (0, 0, sr, 0), mode='replicate')
    else:
        im = F.pad(im[..., -sr:, :], (0, 0, 0, -sr), mode='replicate')
    if sc > 0:
        im = F.pad(im[..., :-sc], (sc, 0, 0, 0), mode='replicate')
    else:
        im = F.pad(im[..., -sc:], (0, -sc, 0, 0), mode='replicate')
    if dim == 2:
        im = im[0, 0]
    elif dim == 3:
        im = im[0]

    return im


def threshold(x, thr):
    '''
    Soft-thresholding operator
    '''

    return F.relu(x - thr) - F.relu(-x - thr)
