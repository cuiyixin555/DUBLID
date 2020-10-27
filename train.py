#!/usr/bin/env python
import os
import torch
import numpy as np
import parameters as pt
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torchvision.utils import make_grid
from network import Network, compute_cost
from loader import BlurredImageDataset, parse
from datetime import datetime
from glob import glob
from os.path import getctime
from tensorboardX import SummaryWriter


def log_params(param_list, var_name, writer, step):
    for i, param in enumerate(param_list):
        var = param.detach().cpu().data.numpy().squeeze()
        writer.add_histogram('%s_%02d' % (var_name, i), var, step)
        grad_val = np.absolute(param.grad.cpu().data.numpy().squeeze())
        writer.add_histogram('grad_%s_%02d' % (var_name, i),
                             np.log(grad_val + 1e-10), step)


def log_image(image, name, writer, step, num_channels=1):
    H, W = image.shape[-2], image.shape[-1]
    image = image.detach().reshape(-1, num_channels, H, W)
    image_grid = make_grid(image, nrow=int(np.sqrt(len(image))),
                           scale_each=True, normalize=True)
    writer.add_image(name, image_grid, step)


if __name__ == '__main__':
    # Initialization
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = pt.logdir_train_root + now + "/"
    # train_dataset = SyntheticDataset(pt.train_image_dir, pt.train_kernel_dir)
    train_dataset = BlurredImageDataset(pt.train_data_dir)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=pt.batch_size, pin_memory=True,
                              num_workers=pt.num_threads)
    test_dataset = BlurredImageDataset(pt.test_image_dir)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1,
                             pin_memory=True, num_workers=pt.num_threads)
    if not os.path.exists(pt.checkpoint_dir):
        os.makedirs(pt.checkpoint_dir)

    # Build network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module = Network(device, channels=pt.image_channels)
    net = DataParallel(module)
    minimizer = torch.optim.Adam(net.parameters(), lr=pt.init_learning_rate)
    # minimizer = torch.optim.SGD(net.parameters(), lr=pt.init_learning_rate,
    # momentum=pt.momentum, nesterov=True)
    init_epoch = 0
    scheduler = torch.optim.lr_scheduler.StepLR(minimizer, pt.decay_every,
                                                gamma=pt.decay_rate)
    weight_list = module.weight_list
    # weight_list = torch.nn.ParameterList(list(module.weight_list)
                                         # + list(module.gain_list))

    # Restart from last run
    if pt.restore:
        checkpoint_files = glob(pt.checkpoint_dir + '/*.ckpt')
        # Load the most recent checkpoint file
        save_path = max(checkpoint_files, key=getctime)
        # save_path = pt.checkpoint_dir + '/epoch_012.ckpt'
        print('restoring model from ' + save_path + '...')
        state = torch.load(save_path)
        net.load_state_dict(state['net_state'])
        minimizer.load_state_dict(state['minimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        init_epoch = state['epoch']
        logdir = state['logdir']
        print('done')

    # Write logs
    train_log_dir = logdir + 'train/'
    test_log_dir = logdir + 'test/'

    # Start main loop
    with SummaryWriter(train_log_dir) as train_writer, \
            SummaryWriter(test_log_dir) as test_writer:
        for epoch in range(init_epoch, pt.max_epochs):
            for i, data in enumerate(train_loader):
                iters = epoch*len(train_loader) + i
                if iters % pt.test_every == 0 and pt.validate:  # Testing step
                    for j, data in enumerate(test_loader):
                        blurred, image, kernel = parse(data, device)
                        step = iters + j
                        write_flag = True if j % pt.write_every == 0 else False
                        with torch.no_grad():
                            image_pred, kernel_pred = net(blurred)
                            loss_val = compute_cost(image_pred, image,
                                                    kernel_pred, kernel,
                                                    weight_list).item()
                            print("[sample (%03d/%03d)] testing loss : %.4f\t"
                                  % (j + 1, len(test_loader), loss_val))
                            test_writer.add_scalar('loss', loss_val, step)
                        # Log to tensorboard
                        if write_flag:
                            log_image(blurred, 'blurred_image',
                                      test_writer, step, blurred.shape[1])
                            log_image(image, 'true_image',
                                      test_writer, step, image.shape[1])
                            log_image(kernel, 'true_kernel', test_writer, step)
                            log_image(image_pred, 'estimated_image',
                                      test_writer, step,
                                      num_channels=image_pred.shape[1])
                            log_image(kernel_pred, 'estimated_kernel',
                                      test_writer, step)
                else:  # Training step
                    minimizer.zero_grad()
                    write_flag = True if i % pt.write_every == 0 else False
                    blurred, image, kernel = parse(data, device)
                    image_pred, kernel_pred = net(blurred)
                    loss = compute_cost(image_pred, image, kernel_pred, kernel,
                                        weight_list)
                    loss.backward()
                    # if loss.detach().item() > 0.25:
                        # import ipdb; ipdb.set_trace()
                        # continue
                    # Gradient clipping
                    max_grad = pt.theta / scheduler.get_lr()[0]
                    torch.nn.utils.clip_grad_value_(net.parameters(), max_grad)
                    minimizer.step()
                    scheduler.step(epoch)
                    # Projection to the positive set
                    with torch.no_grad():
                        for var in module.bias_list:
                            var.data.relu_()
                        for var in module.kernel_bias_list:
                            var.data.relu_()
                        for var in module.prox_list:
                            var.data.relu_()
                    print("[epoch %3.4f] training loss : %.4f\t"
                          % (float(iters) / len(train_loader), loss.item()))
                    # Log to tensorboard
                    train_writer.add_scalar('loss', loss.item(), iters)
                    train_writer.add_scalar('learning_rate',
                                            scheduler.get_lr()[0], iters)
                    if write_flag:
                        log_params(module.weight_list, 'weight',
                                   train_writer, iters)
                        log_params(module.bias_list, 'bias',
                                   train_writer, iters)
                        log_params(module.kernel_bias_list, 'kernel_bias',
                                   train_writer, iters)
                        log_params(module.kernel_prox_list, 'kernel_prox',
                                   train_writer, iters)
                        log_params(module.prox_list, 'prox',
                                   train_writer, iters)
                        log_image(blurred, 'blurred_image',
                                  train_writer, iters, blurred.shape[1])
                        log_image(image, 'true_image',
                                  train_writer, iters, image.shape[1])
                        log_image(image_pred, 'estimated_image', train_writer,
                                  iters, num_channels=image_pred.shape[1])
                        log_image(kernel, 'true_kernel', train_writer, iters)
                        log_image(kernel_pred, 'estimated_kernel',
                                  train_writer, iters)
            # Save intermediate variables at each epoch
            save_path = pt.checkpoint_dir + '/epoch_%03d.ckpt' % epoch
            torch.save({'epoch': epoch, 'logdir': logdir,
                        'net_state': net.state_dict(),
                        'minimizer_state': minimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict()}, save_path)
            print('saving model parameters to ' + save_path)
