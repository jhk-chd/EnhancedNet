# EnhancedNet

This repository is the official Tensorflow implementation for our unpublished paper：

>  EnhancedNet, an end-to-end network for dense disparity estimation and its application to aerial images

and our published paper

> Context pyramidal network for stereo matching regularized by disparity gradients

## Installation

The code is implemented with `Python(3.8)` and `Tensorflow(1.15.5)` for `CUDA Version 12.0`

## Usage

### Inference

1. Donwload the pre-trained model and put it into the 'pre-trained' folder

> pre-trained model donwload link:

2. Run the inference.py with

```
python inference.py --left_path your_left_image_path --right_path your right_image_path --pretrain_model your_pretrain_model_path --net_type <initial or enhanced> --save_dir your_save_dir
```

3. The resulting disparity maps are written to the save folder.


### Citation

Please cite our paper if you use this code or any of the models:

```
@article{kang2019context,
  title={Context pyramidal network for stereo matching regularized by disparity gradients},
  author={Kang, Junhua and Chen, Lin and Deng, Fei and Heipke, Christian},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={157},
  pages={201--215},
  year={2019},
  publisher={Elsevier}
}
```
