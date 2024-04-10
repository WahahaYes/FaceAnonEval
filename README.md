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
10. [FAQ](#faq)
11. [Installation Troubleshooting](#installation-troubleshooting)
12. [Integration with CI/CD](#integration-with-ci-cd)
13. [Performance Metrics](#performance-metrics)
14. [Contact Information](#contact-information)
15. [License](#license)

## Introduction

Privacy concerns related to facial recognition have become a critical aspect of research and development. This framework aims to address these concerns by providing a set of tools for anonymizing facial recognition datasets. It also includes evaluation methodologies to assess the performance of facial recognition systems while maintaining privacy.

## Features

- Anonymization Mechanisms: 
   - [dcos_metric_privacy](src/privacy_mechanisms/dcos_metric_privacy.py): A mechanism for Differential Coefficient of Sensitivity (DCoS) Metric Privacy.
   - [detect_face_mechanism](src/privacy_mechanisms/detect_face_mechanism.py): A mechanism for detecting and anonymizing faces.
   - [gaussian_blur](src/privacy_mechanisms/gaussian_blur.py): Applies Gaussian blur to anonymize facial features.
   - [identity_dp](src/privacy_mechanisms/identity_dp.py): Implements Identity Differential Privacy (IDP) for privacy preservation.
   - [metric_privacy](src/privacy_mechanisms/metric_privacy.py): Implements various metric privacy techniques.
   - [pixel_dp](src/privacy_mechanisms/pixel_dp.py): Applies Pixel Differential Privacy (PixelDP) to images.
   - [simple_mustache](src/privacy_mechanisms/simple_mustache.py): Adds a simple mustache overlay to anonymize faces.
   - [simswap](src/privacy_mechanisms/simswap.py): Applies face swapping for privacy preservation.

- Evaluation Methodologies: 
   - [emotion_utility](src/evaluation/emotion_utility.py): Evaluates the utility of anonymized datasets based on emotion recognition.
   - [lfw_validation](src/evaluation/lfw_validation.py): Evaluates anonymized datasets using the Labeled Faces in the Wild (LFW) dataset.
   - [rank_k](src/evaluation/rank_k.py): Evaluates anonymized datasets using a rank-k evaluation method.
   - [validation_evaluation](src/evaluation/validation_evaluation.py): Performs validation evaluation of anonymized datasets.

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
pip install -r requirements.txt

By default, the requirements are configured to support CUDA-enabled GPUs.  If running on CPU, run the following commands after installing requirements:
pip uninstall onnxruntime-gpu
pip install onnxruntime

We additionally provide a Dockerfile and docker-config.yaml to automatically create an environment.  To create an image and open a terminal within:
docker-compose run --rm faceanoneval

If on Windows, to have full terminal functionality:
winpty docker-compose run --rm faceanoneval
A CPU or GPU targeted image can be specified in the 'docker-compose.yaml' file.

## Usage

### Processing a Dataset

To process a dataset with a specific privacy mechanism, use the following command:
python process_dataset.py --dataset <dataset_name> --privacy_mechanism <mechanism_name>

### Evaluating Anonymized Datasets

To evaluate an anonymized dataset using a specific evaluation methodology, use the following command:
python evaluate_mechanism.py --dataset <dataset_name> --privacy_mechanism <mechanism_name> --evaluation_method <method_name>

## Contributing

Contributions are welcome! To contribute, please follow these guidelines:
- Follow coding standards and conventions used in the project.
- Write clear and concise commit messages.
- Before submitting a pull request, ensure all tests pass and add relevant tests if necessary.

## FAQ

- **Q:** How do I install the framework?
  - **A:** Please refer to the [Installation](#installation) section for detailed instructions on installing the framework and its dependencies

.

- **Q:** What datasets are supported for evaluation?
  - **A:** While we primarily evaluate on CelebA and LFW datasets, some evaluation methodologies accept any identity-labeled dataset. See [Customization](#customization) for more details.

## Installation Troubleshooting

If you encounter any issues during the installation process, please try the following troubleshooting steps:
- Ensure all dependencies are correctly installed by running `pip install -r requirements.txt`.
- If using a GPU, make sure the CUDA toolkit and cuDNN are properly configured.
- If running into compatibility issues, check for updates or patches for the dependencies.

## Integration with CI/CD

The framework can be integrated into continuous integration/continuous deployment (CI/CD) pipelines for automated testing and deployment. Detailed instructions for integration with popular CI/CD tools will be provided in the documentation soon.

## Performance Metrics

Performance metrics and benchmarks for the privacy mechanisms and evaluation methodologies will be included in future releases to help users understand their effectiveness and performance characteristics in different scenarios.

## Contact Information

For any questions, issues, or feedback, please feel free to reach out to us via [GitHub Issues](https://github.com/your-repo-name/issues) or [Support Forum](https://example.com/support-forum). We value your input and are committed to providing timely assistance and support.

Here are the citations:

@inproceedings{DBLP:conf/mm/ChenCNG20,
  author    = {Renwang Chen and
               Xuanhong Chen and
               Bingbing Ni and
               Yanhao Ge},
  title     = {SimSwap: An Efficient Framework For High Fidelity Face Swapping},
  booktitle = {{MM} '20: The 28th {ACM} International Conference on Multimedia},
  year      = {2020},
  url       = {https://dl.acm.org/doi/10.1145/3394171.3413543}
}

```bixtex
@inproceedings{deng2020subcenter,
  title     = {Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces},
  author    = {Deng, Jiankang and Guo, Jia and Liu, Tongliang and Gong, Mingming and Zafeiriou, Stefanos},
  booktitle = {Proceedings of the IEEE Conference on European Conference on Computer Vision},
  year      = {2020},
  url       = {https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_Sub-center_ArcFace_Boosting_Face_Recognition_by_Large-scale_Noisy_Web_CVPR_2020_paper.html}
}

@inproceedings{Savchenko_2022_CVPRW,
  author    = {Savchenko, Andrey V.},
  title     = {Video-Based Frame-Level Facial Analysis of Affective Behavior on Mobile Devices Using EfficientNets},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2022},
  pages     = {2359-2366},
  url       = {https://arxiv.org/abs/2103.17107}
}

@article{xue2023face,
  title     = {Face image de-identification by feature space adversarial perturbation},
  author    = {Xue, Hanyu and Liu, Bo and Yuan, Xin and Ding, Ming and Zhu, Tianqing},
  journal   = {Concurrency and Computation: Practice and Experience},
  volume    = {35},
  number    = {5},
  pages     = {e7554},
  year      = {2023},
  publisher = {Wiley Online Library},
  url       = {https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.7554}
}

@inproceedings{richardson2021encoding,
  author    = {Richardson, Elad and Alaluf, Yuval and Patashnik, Or and Nitzan, Yotam and Azar, Yaniv and Shapiro, Stav and Cohen-Or, Daniel},
  title     = {Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2021},
  url       = {https://openaccess.thecvf.com/content/CVPR2021/html/Richardson_Encoding_in_Style_A_StyleGAN_Encoder_for_Image-to-Image_Translation_CVPR_2021_paper.html}
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
```
