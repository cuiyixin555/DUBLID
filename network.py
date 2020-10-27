import torch
import torch.nn as nn
import torch.nn.functional as F
import operations as op
import parameters as pt
from math import ceil


class Network(nn.Module):
    def __init__(self, device=torch.device('cpu'), channels=1):
        super(Network, self).__init__()
        self.device = device
        weight = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(pt.C, channels, pt.K, pt.K, device=self.device)))
        self.weight_list = nn.ParameterList([weight])
        for layer in range(pt.num_layer - 1):
            weight = nn.Parameter(nn.init.xavier_normal_(
                torch.empty(pt.C, pt.C, 3, 3, device=self.device)))
            self.weight_list.append(weight)
        self.bias_list = nn.ParameterList()
        for layer in range(pt.num_layer + 2):
            # Rectifier thresholds
            bias = nn.Parameter(torch.full(size=(1, pt.C, 1, 1), device=device,
                                           fill_value=pt.bias_init))
            self.bias_list.append(bias)
        self.kernel_bias_list = nn.ParameterList()
        for layer in range(pt.num_layer):
            kernel_bias = nn.Parameter(torch.full(
                size=(1,), device=device, fill_value=pt.kernel_bias_init))
            self.kernel_bias_list.append(kernel_bias)
        self.kernel_prox_list = nn.ParameterList()
        for layer in range(pt.num_layer):
            kernel_prox = nn.Parameter(torch.full(
                size=(1,), device=device, fill_value=pt.kernel_prox_init))
            self.kernel_prox_list.append(kernel_prox)
        self.prox_list = nn.ParameterList()
        for layer in range(pt.num_layer):
            # Proximity to the previous solutions
            zeta = nn.Parameter(torch.full(size=(1, pt.C, 1, 1), device=device,
                                           fill_value=pt.zeta_init))
            self.prox_list.append(zeta)
        eta = nn.Parameter(torch.full(size=(1, pt.C, 1, 1), device=device,
                                      fill_value=pt.eta_init))
        self.prox_list.append(eta)

    def _reflect_filter(self, w):
        '''
        Symmetric reflection of filters arount the origin
        Input:
            w:  CoutxCinxHkxWk
        '''

        Cout, Cin, Hk, Wk = w.size()
        w_ref = torch.reshape(w, shape=(-1, Hk*Wk))
        reverse_ind = torch.arange(Hk*Wk - 1, -1, -1, device=self.device)
        w_ref = torch.index_select(w_ref, 1, reverse_ind)
        w_ref = torch.reshape(w_ref, shape=(Cout, Cin, Hk, Wk))

        return w_ref

    def _compute_image_coeffs(self, Fy, Fg, Fw, Fk):
        '''
        Solve for image coefficients from feature maps and kernel
        Input:
          Fy:       NxC_inxHfxWf2x2; fourier coefficients of blurred images
          Fg:       NxCxHfxWf2x2; fourier coefficients of feature maps
          Fw:       C_inxCxHfxWf2x2 where C_in=1 for grayscale images
                    and C_in=3 for color images; fourier coefficients of
                    the weights in the first feature extraction layer
          Fk:       NxHfxWf2x2;  fourier coefficients of blur kernels
        Output:
          Fx:       NxHfxWf2x2; fourier coefficients of estimated images
        '''

        eta = pt.prox_scale*self.prox_list[-1]
        ec = eta.unsqueeze(dim=-1)
        if Fw.shape[0] == 1:  # grayscale
            Fy = Fy[:, 0]
            num = op.conj_mul(Fk, Fy) \
                + torch.sum(ec*op.conj_mul(Fw, Fg), dim=1)
            den = op.csquare(Fk) + torch.sum(eta*op.csquare(Fw), dim=1)
            Fx = num / den.unsqueeze(dim=-1)
            Fx = Fx.unsqueeze(dim=1)
        elif Fw.shape[0] == 3:  # color
            Fwr = Fw[0].unsqueeze(dim=0)
            Fwg = Fw[1].unsqueeze(dim=0)
            Fwb = Fw[2].unsqueeze(dim=0)
            Fyr, Fyg, Fyb = Fy[:, 0], Fy[:, 1], Fy[:, 2]
            Crr = op.csquare(Fk) + torch.sum(eta*op.csquare(Fwr), dim=1)
            Cgg = op.csquare(Fk) + torch.sum(eta*op.csquare(Fwg), dim=1)
            Cbb = op.csquare(Fk) + torch.sum(eta*op.csquare(Fwb), dim=1)
            Crg = torch.sum(ec*op.conj_mul(Fwr, Fwg), dim=1)
            Crb = torch.sum(ec*op.conj_mul(Fwr, Fwb), dim=1)
            Cgb = torch.sum(ec*op.conj_mul(Fwg, Fwb), dim=1)
            Br = op.conj_mul(Fk, Fyr) \
                + torch.sum(ec*op.conj_mul(Fwr, Fg), dim=1)
            Bg = op.conj_mul(Fk, Fyg) \
                + torch.sum(ec*op.conj_mul(Fwg, Fg), dim=1)
            Bb = op.conj_mul(Fk, Fyb) \
                + torch.sum(ec*op.conj_mul(Fwb, Fg), dim=1)
            Irr = Cgg*Cbb - op.csquare(Cgb)
            Igg = Crr*Cbb - op.csquare(Crb)
            Ibb = Crr*Cgg - op.csquare(Crg)
            Irg = op.conj_mul(Cgb, Crb) - op.real_mul(Cbb, Crg)
            Irb = op.mul(Crg, Cgb) - op.real_mul(Cgg, Crb)
            Igb = op.conj_mul(Crg, Crb) - op.real_mul(Crr, Cgb)
            den = Crr*(Cgg*Cbb - op.csquare(Cgb)) \
                - Cgg*op.csquare(Crb) - Cbb*op.csquare(Crg) \
                + 2*op.real(op.conj_mul(op.mul(Crg, Cgb), Crb))
            Fxr = op.real_mul(Irr, Br) + op.mul(Irg, Bg) + op.mul(Irb, Bb)
            Fxg = op.conj_mul(Irg, Br) + op.real_mul(Igg, Bg) + op.mul(Igb, Bb)
            Fxb = op.conj_mul(Irb, Br) + op.conj_mul(Igb, Bg) \
                + op.real_mul(Ibb, Bb)
            Fx = torch.stack([Fxr, Fxg, Fxb], dim=1)
            Fx /= den.unsqueeze(dim=1).unsqueeze(dim=-1)

        return Fx

    def forward(self, blurred_image):
        '''
        The main deblurring network
        Input:
          blurred_image:    NxC_inxHfxWf
        Output:
          image_pred:       NxC_inxHvxWv; estimated image features
          kernel_pred:      NxHkxWk; estimated kernels
        '''

        # Size for kernels
        Hk, Wk = pt.bounding_box_size
        # Size for the 'full' scheme
        N, C_in, Hv, Wv = blurred_image.size()
        # Size for the 'same' scheme
        Hs, Ws = Hv + Hk - 1, Wv + Wk - 1
        # Size for the 'valid' scheme
        Hb, Wb = Hs + Hk - 1, Ws + Wk - 1

        # Feature extraction: filter the blurred images
        wy_list = []
        fy = blurred_image
        # Pad for the maximum possible filter size
        fft_size = (int(ceil(Hb / 64.) * 64), int(ceil(Wb / 64.) * 64))
        for layer in range(pt.num_layer):
            w = self.weight_list[layer]
            if layer == 0:
                w_mean = torch.mean(w.view(pt.C, C_in, -1), dim=-1)
                w = w - torch.reshape(w_mean, (pt.C, C_in, 1, 1))
                w0 = torch.transpose(w, dim0=0, dim1=1)
                # At the output end we need to perform convolution instead
                # of correlation as we rely on the convolution theorem
                # for image reconstruction using FFT
                w = self._reflect_filter(w)
            fy = op.conv2(fy, w)
            fy_padded = op.pad_to(
                F.pad(fy, pad=(Wk - 1, Wk - 1, Hk - 1, Hk - 1)), size=fft_size)
            wy_list.append(fy_padded)

        # Deconvolution: estimate the kernel
        delta = torch.zeros((N, Hk, Wk), device=self.device)
        delta[:, Hk//2, Wk//2] = 1
        b0 = self.bias_list[0]
        z = op.threshold(op.circ_shift(wy_list[-1], (Hk//2, Wk//2)), b0)
        Fz = op.fft2(z)
        k = delta
        for layer in range(pt.num_layer):
            # Retrieve filtered blurred image
            fy = wy_list.pop()
            Ffy = op.fft2(fy)
            Fk = op.fft2(k, size=fft_size).unsqueeze(dim=1)
            fft_size = fy.shape[-2:]
            # Update latent image
            zeta = pt.prox_scale*self.prox_list[layer]
            num = zeta.unsqueeze(dim=-1)*op.conj_mul(Fk, Ffy) + Fz
            den = zeta*op.csquare(Fk) + 1
            Fg = num / den.unsqueeze(dim=-1)
            # Update surrogate blurred image
            b = self.bias_list[layer + 1]
            Fz = op.fft2(op.threshold(op.ifft2(Fg, size=fft_size), b))
            # Update kernels
            zk = self.kernel_prox_list[layer]
            num = zk*torch.sum(op.conj_mul(Fz, Ffy), dim=1) + Fk.squeeze(dim=1)
            den = zk*torch.sum(op.csquare(Fz), dim=1) + 1
            k = op.ifft2(num / den.unsqueeze(dim=-1), size=fft_size)
            k_max = torch.logsumexp(k.view(N, -1)*pt.kernel_scale, dim=-1)
            k_max = k_max / pt.kernel_scale
            bk =  pt.kernel_bias_scale*self.kernel_bias_list[layer]
            k = F.relu(k[:, :Hk, :Wk] - bk*torch.reshape(k_max, (N, 1, 1)))
            k_sum = k.sum(1, keepdim=True).sum(2, keepdim=True)
            k = (k + pt.epsilon*delta) / (k_sum + pt.epsilon)

        # Reconstruct image from feature map
        Fy = op.fft2(F.pad(blurred_image,
                           pad=(Wk - 1, Wk - 1, Hk - 1, Hk - 1)),
                     size=fft_size)
        Fk = op.fft2(k, size=fft_size)
        Fw0 = op.fft2(op.circ_shift(op.pad_to(w0, size=fft_size),
                                    (pt.K//2, pt.K//2)))
        b = self.bias_list[pt.num_layer + 1]
        Fg = op.fft2(op.threshold(op.ifft2(Fg, size=fft_size), b))
        Fx = self._compute_image_coeffs(Fy, Fg, Fw0, Fk)
        image = op.ifft2(Fx, size=fft_size)
        # Only the interior can be reliably recovered
        image = image[:, :, Hk//2:Hk//2 + Hv, Wk//2:Wk//2 + Wv]
        kernel = k

        return image, kernel
