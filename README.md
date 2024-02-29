# Facial Recognition Anonymization Framework

## Overview

The Facial Recognition Anonymization Framework is a powerful tool designed for researchers and developers working in the field of privacy-preserving facial recognition. This framework provides a set of utilities, privacy mechanisms, and evaluation methodologies for processing and evaluating facial recognition datasets while preserving the privacy of individuals.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Processing a Dataset](#processing-a-dataset)
   - [Evaluating Anonymized Datasets](#evaluating-anonymized-datasets)
6. [Privacy Mechanisms](#privacy-mechanisms)
7. [Evaluation Methodologies](#evaluation-methodologies)
8. [Customization](#customization)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

Privacy concerns related to facial recognition have become a critical aspect of research and development. This framework aims to address these concerns by providing a set of tools for anonymizing facial recognition datasets. It also includes evaluation methodologies to assess the performance of facial recognition systems while maintaining privacy.

## Features

- Anonymization Mechanisms: 
   - Gaussian Blur
   - Uniform Blur
   - Pixel DP
   - Metric Privacy (SVD)
- Evaluation Methodologies: 
   - Rank-k Evaluation
   - Validation Evaluation
   - LFW (Labeled Faces in the Wild) Validation Evaluation
- Customizable: Easily extendable with new privacy mechanisms and evaluation methodologies
- Batch Processing: Efficiently processes datasets in batches for scalability
- Detailed Documentation: Comprehensive documentation for each module and functionality
- Open Source: Released under an open-source license for collaboration and contributions

## Requirements

- Python 3.10
- Additional requirements are specified in `requirements.txt`.
- Download the CelebA dataset (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place at 'Datasets/CelebA/'.
- Download the Labeled Faces in the Wild (LFW) dataset (https://vis-www.cs.umass.edu/lfw/) and place at 'Datasets/lfw' (we use the unaligned images, as our codebase performs face detection/ alignment).
- While we evaluate on CelebA and lfw, some evaluation methodologies will accept any identity-labeled dataset.  See [Customization](#customization) for more details.

## Installation

Install requirements with:
```
pip install -r requirements.txt
```

By default, the requirements are configured to support CUDA-enabled GPUs.  If running on CPU, run the following commands after installing requirements (these lines can additionally be uncommented from the Dockerfile to automatically generate CPU-ready Docker images):
```
pip uninstall onnxruntime-gpu
pip install onnxruntime
```

We additionally provide a Dockerfile and docker-config.yaml to automatically create an environment.  To create an image and open a terminal within:
```
docker-compose run --rm faceanoneval
```

If on Windows, to have full terminal functionality:
```
winpty docker-compose run --rm faceanoneval
```

