# DDcGAN for infrared and visible image fusion in TensorFlow 2
This repository contains the code for the thesis ‘Fantastic Privacy Techniques and How to Implement Them: Navigating Differential Privacy in Multi-Modal Generative Models for Clinical Data.’ 

This repository contains an implementation of the Dual-Discriminator Conditional Generative Adversarial Network (DDcGAN) using TensorFlow 2 and with added Differential Privacy Techniques. DDcGAN is a powerful framework for multi-modal image fusion, particularly suited for infrared-visible and MRI-PET image pairs. The generator combines visible and infrared images into a single fused image, while two discriminators (D1 and D2) ensure that the generated images resemble the original images.

This implementation is based on the original [DDcGAN paper](https://ieeexplore.ieee.org/abstract/document/9031751).

The following datasets are used in this project:
- [Training_Dataset.h5](https://drive.google.com/file/d/1o-dhSphyyiqSHu9veiKWvxViZ_FSeZWJ/view?usp=share_link) - infrared and visible images from TNO Human Factors Dataset. 

## Features
- Implemented in TensorFlow 2
- Supports multi-resolution image fusion
- Includes data preprocessing and augmentation
- Provides evaluation metrics for generated images

## Installation

```
git clone https://github.com/yourusername/ddcgan-vis-ir.git
cd ddcgan-vis-ir
```
## **Requirements**

To run this project, you need the following dependencies:

- Python 3.x
- TensorFlow 2.x
- NumPy
- h5py
- SciPy
- Matplotlib
- Imageio
- OpenCV

Install the required libraries using `pip`:

```bash
pip install tensorflow numpy h5py scipy matplotlib imageio opencv-python
```

# Base Model Usage
To train the model with no privacy guarantees on your dataset, run the main.py script with the IS_TRAINING = True flag:
```
python main.py
```
This will load the training dataset (Training_Dataset.h5), train the model for a specified number of epochs and batch size, and save the model checkpoints in the ./model/ directory.
You can adjust the training parameters (batch size, number of epochs, logging intervals) inside the main.py file.

Once the model is trained, you can generate fused images using the trained model. Set the IS_TRAINING = False flag in main.py and run:
```
python main.py
```
This will load the pre-trained model checkpoints from the ./model/ directory and generate fused images for the test set (e.g., VIS.bmp and IR.bmp images) located in the ./test_imgs/ directory.
If you want to change the number of generated images or paths for the input images, modify the main.py script

# Data Privacy
Three main techniques for achieving data privacy have been implemented in this project: input perturbation, intermediate perturbation, and output perturbation.

## Input Perturbation
To train the model with input perturbation on your dataset, run the main_input.py script with the IS_TRAINING = True flag:
```
python main_input.py
```
You can change noise type ('gaussian' or 'laplace') and epsilon (privacy guarantee) value.

