#!/usr/bin/env python3
import torch
import time
import parameters as pt
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from glob import glob
from os.path import getctime
from loader import BlurredImageDataset, imwrite, parse
from skimage.morphology import remove_small_objects
from network import Network


def to_numpy_image(tensor):
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3:
        array = array.transpose(1, 2, 0)
        if array.shape[2] == 1:
            array = array[:, :, 0]
    # Clip image to [0, 1]
    array[array < 0] = 0
    array[array > 1] = 1

    return array


if __name__ == '__main__':
    # Feed testing data
    torch.cuda.empty_cache()
    dataset = BlurredImageDataset(pt.test_image_dir)
    loader = DataLoader(dataset, shuffle=False, batch_size=1,
                        num_workers=pt.num_threads)

    # Restore saved models
    with torch.no_grad():
        checkpoint_files = glob(pt.checkpoint_dir + '/*.ckpt')
        # Load the most recent checkpoint file
        save_path = max(checkpoint_files, key=getctime)
        print('restoring model from ' + save_path + '...')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            state = torch.load(save_path)
        else:
            device = torch.device('cpu')
            state = torch.load(save_path, map_location='cpu')
        module = Network(device, channels=pt.image_channels)
        net = DataParallel(module)
        net.load_state_dict(state['net_state'])
        print('done')

        # Iterate over test images
        for i, data in enumerate(loader):
            # Get outputs
            blurred, image, kernel = parse(data, device)
            start = time.time()
            im_est, ker_est = net(blurred)
            end = time.time()
            im_est = to_numpy_image(im_est[0])
            ker_est = to_numpy_image(ker_est[0])
            ker_est /= ker_est.max()
            ker_est *= remove_small_objects(ker_est > 0, min_size=8)
            out_file = '%05d.png' % i
            imwrite(im_est, pt.result_dir + 'images/' + out_file)
            imwrite(ker_est, pt.result_dir + 'kernels/' + out_file)
            print('Finished processing ' + out_file)
