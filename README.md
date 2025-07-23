# Facial Recognition Anonymization Framework

## Overview

This codebase is a generation and evaluation pipeline for privacy-preserving neural face synthesis.  A collection of face synthesis algorithms designed to obfuscate an input face's underlying identity are developed.  A suite of evaluation algorithms, including facial recognition and age, race, and gender classification networks, are implemented to evaluate the efficacy of each privacy mechanism.  Additional privacy mechanisms can easily be included for evaluation.  This presents a comprehensive framework for implementing and evaluating privatized faces against a wide set of benchmarks on a standardized evaluation pipeline.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Processing a Dataset](#processing-a-dataset)
   - [Evaluating Anonymized Datasets](#evaluating-anonymized-datasets)
6. [Contributing](#contributing)
7. [Citations](#citations)
8. [License](#license)

## Introduction

Privacy concerns related to facial recognition have become a critical aspect of research and development. This framework aims to address these concerns by providing a set of tools for anonymizing facial recognition datasets. It also includes evaluation methodologies to assess the performance of facial recognition systems while maintaining privacy.

## Features

- Anonymization Mechanisms: 
   - [dtheta_privacy](src/privacy_mechanisms/dtheta_privacy_mechanism.py): A mechanism for applying the *AvatarLDP* and *AvatarRotation* mechanisms, which rotate identity embeddings over the embedding hypersphere to a requested level of dissimilarity, to faces (Wilson et al. 2025).
   - [gaussian_blur](src/privacy_mechanisms/gaussian_blur_mechanism.py): Applies Gaussian blur to anonymize facial features.
   - [uniform_blur](src/privacy_mechanisms/uniform_blur_mechanism.py): Applies uniform blur to anonymize facial features.
   - [identity_dp](src/privacy_mechanisms/identity_dp_mechanism.py): Implements Identity DP for privacy preservation using the SimSwap face swapping architecture as a backend.
   - [metric_privacy](src/privacy_mechanisms/metric_privacy_mechanism.py): Implements metric privacy to faces, MetricSVD (Fan 2019).
   - [pixel_dp](src/privacy_mechanisms/pixel_dp_mechanism.py): Applies Pixel Differential Privacy (PixelDP) to images (Fan 2018).
   - [simswap](src/privacy_mechanisms/simswap_mechanism.py): Applies face swapping for privacy preservation (Chen et al. 2020).

- Evaluation Methodologies: 
   - [utility](src/evaluation/utility/utility_evaluation.py): Evaluates the utility of anonymized datasets based on age, race, gender, and emotion recognition.
   - [lfw_validation](src/evaluation/lfw_validation_evaluation.py): Evaluates validation / EER of anonymized datasets using the Labeled Faces in the Wild (LFW) dataset.
   - [rank_k](src/evaluation/rank_k_evaluation.py): Evaluates anonymized datasets using a rank-k evaluation method.
   - [validation_evaluation](src/evaluation/validation_evaluation.py): Performs validation / EER evaluation of anonymized datasets.

- Customizable: Easily extendable with new privacy mechanisms and evaluation methodologies.
- Batch Processing: Efficiently processes datasets in batches for scalability.
- Detailed Documentation: Each module and functionality contains documentation (in progress).
- Open Source: Released under an open-source license for collaboration and contributions.

## Requirements

- Python 3.10
- Additional requirements are specified in `requirements.txt`.
- Download the CelebA dataset (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place at 'Datasets/CelebA/'.
- Download the Labeled Faces in the Wild (LFW) dataset (https://vis-www.cs.umass.edu/lfw/) and place at 'Datasets/lfw' (we use the unaligned images, as our codebase performs face detection/ alignment).
- While we evaluate on CelebA and lfw, some evaluation methodologies will accept any identity-labeled dataset.  See [Customization](#customization) for more details.

## Installation

Install requirements with:
`pip install -r requirements.txt`

By default, the requirements are configured to support CUDA-enabled GPUs.  If running on CPU, run the following commands after installing requirements:
`pip uninstall onnxruntime-gpu`
`pip install onnxruntime`

We additionally provide a Dockerfile and docker-config.yaml to automatically create an environment.  To create an image and open a terminal within:
`docker-compose run --rm faceanoneval`

If on Windows, to have full terminal functionality:
`winpty docker-compose run --rm faceanoneval`

A CPU or GPU targeted image can be specified in the 'docker-compose.yaml' file.

## Usage

### Processing a Dataset

To process a dataset with a specific privacy mechanism, use the following command:
`python process_dataset.py --dataset <dataset_name> --privacy_mechanism <mechanism_name>`

### Evaluating Anonymized Datasets

To evaluate an anonymized dataset using a specific evaluation methodology, use the following command:
`python evaluate_mechanism.py --dataset <dataset_name> --privacy_mechanism <mechanism_name> --evaluation_method <method_name>`

## Contributing

Contributions are welcome! To contribute, please follow these guidelines:
- Follow coding standards and conventions used in the project.
- Write clear and concise commit messages.
- Fork this repo and submit a pull request.

For any questions, issues, or feedback, please feel free to reach out post an [Issue](https://github.com/your-repo-name/issues).

## Citations

Here are the citations for implemented algorithms:
```bibtex
# TODO: Wilson et al 2025

@inproceedings{chen_simswap_2020,
  author    = {Renwang Chen and
               Xuanhong Chen and
               Bingbing Ni and
               Yanhao Ge},
  title     = {SimSwap: An Efficient Framework For High Fidelity Face Swapping},
  booktitle = {{MM} '20: The 28th {ACM} International Conference on Multimedia},
  year      = {2020},
  url       = {https://dl.acm.org/doi/10.1145/3394171.3413543}
}


@inproceedings{deng_arcface_2019,
  title    = {ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author   = {Deng, 
               Jiankang and 
               Guo, 
               Jia and 
               Niannan, 
               Xue and 
               Zafeiriou, 
               Stefanos},
  booktitle = {CVPR},
  year      = {2019}
}

@article{wen_identitydp_2022,
  title     = {IdentityDP: Differential private identification protection for face images},
  author    = {Wen, Yunqian and Liu, Bo and Ding, Ming and Xie, Rong and Song, Li},
  year      = {2022},
  journal   = {Neurocomput.},
  publisher = {Elsevier Science Publishers B. V.},
  volume    = {501},
  number    = {C},
  month     = {Aug},
  pages     = {197–211},
  doi       = {10.1016/j.neucom.2022.06.039}
}

@inproceedings{fan_practical_2019,
  title={Practical image obfuscation with provable privacy},
  author={Fan, Liyue},
  booktitle={2019 IEEE international conference on multimedia and expo (ICME)},
  pages={784--789},
  year={2019},
  organization={IEEE}
}

@inproceedings{fan_image_2018,
  title={Image pixelization with differential privacy},
  author={Fan, Liyue},
  booktitle={IFIP Annual Conference on Data and Applications Security and Privacy},
  pages={148--162},
  year={2018},
  organization={Springer}
}

```

## License

MIT License

Copyright 2024 Ethan Wilson and Eakta Jain

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
