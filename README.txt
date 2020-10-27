This folder contains a reference implementation of the algorithm proposed in the following paper:
Y. Li, M. Tofighi, J. Geng, V. Monga and Y. C. Eldar, "An Algorithm Unrolling Approach to
Deep Blind Image Deblurring," IEEE Transactions on Image Processing, under review.

If you would like to use the code for any publications, please kindly cite the above reference.

Requirements:
Python (3.7.2)
PyTorch (1.0.1)
Numpy (1.16.2)
Scipy (1.2.1)
Scikit-Image (0.14.2)

Descriptions:
loader.py		Data loading module, including functions for data loading, augmentations, pre-processing, etc.
networks.py		Defining network architecture
test.py			Inference module, which performs the actual work of blind deblurring
parameters.py	Parameter configurations, including network architectures, data paths, etc.
operations.py	Helper module containing miscellaneous functions, such as complex operations, scaling, padding, etc.

Instructions:
1. Install all required python modules listed in the "Requirements" section.
2. Place the network model under 'checkpoints' directory:
	-- A copy of the model learned from linear kernels has already been placed
	-- By default, the most current model will be loaded; to modify it, please edit the 'save_path' variable in the 'test.py' script
3. Prepare the testing data under 'test' folder and edit the paths in 'parameters.py' correspondingly; by default, the dataset is organized as follows:
	test
	  |-- <dataset>
			 |-- blurred: folder containing blurred images
			 |-- kernel: groundtruth kernels, if available
			 |-- sharp: groundtruth sharp images
	Note that the number of images in each subfolder must be equal, and they must be arranged in corresponding order.
4. Execute the 'test.py' script.
5. The deblurred results will be saved to 'results' folder.

Enjoy!

Contact:
If there are any questions, please contact Yuelong Li (liyuelongee@gmail.com)
