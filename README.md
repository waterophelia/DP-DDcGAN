# Differential Privacy in DDcGAN for image fusion in TensorFlow 2
This repository contains the code for the paper ‘Fantastic Privacy Techniques and How to Implement Them: Navigating Differential Privacy in Multi-Modal Generative Models for Clinical Data.’ - an implementation of the Dual-Discriminator Conditional Generative Adversarial Network (DDcGAN) using TensorFlow 2 with added Differential Privacy Techniques. DDcGAN is a powerful framework for multi-modal image fusion, particularly suited for infrared-visible and MRI-PET image pairs. The generator combines visible and infrared images into a single fused image, while two discriminators (D1 and D2) ensure that the generated images resemble the original images.

## Features
- Implemented in TensorFlow 2.17
- Supports multi-resolution visible-infrared image fusion
- Provides evaluation metrics for generated images

This implementation is based on the original [DDcGAN paper](https://ieeexplore.ieee.org/abstract/document/9031751).

The following datasets are used in this project:
- [irvis.h5](https://drive.google.com/file/d/1o-dhSphyyiqSHu9veiKWvxViZ_FSeZWJ/view?usp=share_link) - infrared and visible images from TNO Human Factors Dataset.
- [mripet.h5](https://drive.google.com/file/d/1V2N9ujvqJkIUpzhOe3_TZqdQpICVi-m8/view?usp=sharing) - MRI and PET images from Harvard Medical School Whole Brain Atlas website.

## Installation

```
git clone https://github.com/waterophelia/DP-DDcGAN.git
cd DP-DDcGAN
```
## Requirements

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

## Base Model Usage
To train the model without privacy guarantees on your dataset, comment out noise_type and epsilon in main.py from both hyperparameters and the training loop, similarly delete it from train.py in def train and from generate.py in def generate, and in train.py get rid of applying the noise to both batches; similarly, in generate.py get rid of applying noisy outputs.
Then, run the main.py script with the IS_TRAINING = True flag:
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

# Differential Privacy Techniques
Three main techniques for achieving data privacy have been implemented in this project: input perturbation, intermediate perturbation, and output perturbation. For intermediate perturbation, Functional Mechanism and DP-SGD.

## Input Perturbation
To train the model with input perturbation on your dataset, run the main_input.py script with:
1. noise_type: Set to 'gaussian' or 'laplace' depending on the type of noise you want.
2. epsilon: Set the privacy guarantee for the input data.
```
python main.py
```

## Output Perturbation
To enable output perturbation, adjust the following parameters in main.py:
1. noise_type: Set to 'gaussian' or 'laplace' depending on the type of noise you want.
2. epsilon: Set the privacy guarantee, where lower values provide stronger privacy.
```
python main.py
```
## Functional Mechanism
To train the model with Functional Mechanism (either with Laplace or Gaussian noise), use main_functional.py and train_functional.py instead of the main.py and train.py.
```
python main_functional.py
```

## DP-SGD
To train the model with DP-SGD Mechanism with Moments Accountant, use main_dpsgd.py and train_sdpsgd.py instead of the main.py and train.py.
```
python main_dpsgd.py
```
## Evaluation
Once you’ve trained the model or generated images with differential privacy, you can evaluate the results using the provided evaluation metrics. These metrics can be adjusted inside the evaluation script.
```
python evaluate.py
```

You can also generate comparisons of the generated images and privacy-utility trade-off graphs with files combined_graphics.py and graphs.py.
```
python graphs.py
```

