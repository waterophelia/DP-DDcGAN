# DDcGAN for infrared and visible image fusion in TensorFlow 2

This repository contains an implementation of the Dual-Discriminator Conditional Generative Adversarial Network (DDcGAN) using TensorFlow 2. DDcGAN is a powerful framework for multi-modal image fusion, particularly suited for infrared-visible and MRI-PET image pairs. The generator combines visible and infrared images into a single fused image, while two discriminators (D1 and D2) ensure that the generated images resemble the original images.

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
