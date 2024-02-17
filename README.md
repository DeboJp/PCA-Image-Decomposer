# PCA-Image-Decomposer
PCA Image Decomposer is a Python-based tool that applies Principal Component Analysis (PCA) to a dataset of images. The primary function of this tool is to decompose images into their principal components, enabling dimensionality reduction for image processing tasks. This is particularly useful for image compression, noise reduction, and feature extraction.

## Features

- Load and center image datasets.
- Calculate the covariance matrix for the set of images.
- Perform eigendecomposition to identify principal components.
- Project images onto a lower-dimensional space.
- Reconstruct images from their principal components.
- Visualize original and reconstructed images.

## Prerequisites
- Python 3.x
- NumPy library
- Matplotlib library (for visualization)
- Image dataset in .npy format (e.g., Iris_64x64.npy)

## Installation

Before running the PCA Image Decomposer, ensure you have Python installed on your machine. You can download Python from python.org. After installing Python, install the required libraries using pip:

`pip install numpy matplotlib`

## Downloading the Image Dataset

The image dataset required for PCA should be in NumPy (.npy) format. A sample dataset Iris_64x64.npy is utilized for demonstration purposes. Ensure your dataset is located in the project's root directory.

## How to Run

Clone the repository to your local machine or download the source code.

Navigate to the project directory: `cd "path_to_PCA_Image_Decomposer" `

Run the main script (e.g., main.py) to start the decomposition process: `main.py`

The script will output the reconstructed images using PCA and visualize them using Matplotlib.

## 

The program is particulary useful when working with image data.

- Reducing the dimensions of the image data while retaining the most important information, which is essential for machine learning models to train efficiently.
- Filtering noise from images, enhancing the quality of the dataset.
- Visualizing high-dimensional data in two or three dimensions.

This tool serves as an educational resource for understanding and implementing PCA on image data, as well as a starting point for more complex image processing tasks.
