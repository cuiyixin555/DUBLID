# Parameters for neural network
C = 16  # number of feature maps
K = 3  # filter width
num_layer = 10  # total number of layer
epsilon = 1e-8  # to avoid division by zero
alpha = 1e-2  # weight decay parameter
kappa = 1e2  # weight for the kernel term
bias_init = 0.02
zeta_init = 1.
prox_scale = 10.
kernel_prox_init = 1.
kernel_bias_init = 0.
kernel_scale = 1e2
kernel_bias_scale = 0.01
eta_init = 1.

# Parameters for data loading
batch_size = 16
patch_size = [256, 256]
num_threads = batch_size  # number of threads to use in data loading
image_channels = 1  # 1 for grayscale, and 3 for color
bounding_box_size = [45, 45]
noise_stddev = 0.01

# Parameters for testing
test_dir = 'test'
checkpoint_dir = 'checkpoints/'
dataset_test = 'BSDS'
result_dir = 'results/' + dataset_test + '/'
test_image_dir = test_dir + '/' + dataset_test
validate = False
