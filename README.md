# VTGAN-pytorch-version
The official code written by tensorflow is :
[Official code](https://github.com/SharifAmit/VTGAN)


### Arxiv Pre-print
```
https://arxiv.org/abs/2104.06757
```
### CVF ICCVW 2021
```
https://openaccess.thecvf.com/content/ICCV2021W/CVAMD/html/Kamran_VTGAN_Semi-Supervised_Retinal_Image_Synthesis_and_Disease_Prediction_Using_Vision_ICCVW_2021_paper.html
```
### IEE Xplore ICCVW 2021
```
https://ieeexplore.ieee.org/document/9607858
```
# Citation 
```
@INPROCEEDINGS{9607858,
  author={Kamran, Sharif Amit and Hossain, Khondker Fariha and Tavakkoli, Alireza and Zuckerbrod, Stewart Lee and Baker, Salah A.},
  booktitle={2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)}, 
  title={VTGAN: Semi-supervised Retinal Image Synthesis and Disease Prediction using Vision Transformers}, 
  year={2021},
  volume={},
  number={},
  pages={3228-3238},
  doi={10.1109/ICCVW54120.2021.00362}
}
```

## Pre-requisite
- Ubuntu
- NVIDIA Graphics card
- torch 1.12.0
- torchvision 0.13.0 
- cu116
- visdom 0.1.8.9

Other pre-requisite like the PYYAML or scipy are connected to the package above, 
if the program goes wrong, you can check the  report and install them.

## Dataset
You can find the download link in the official code website

## Effect
This is the effect after the models have been trained for 35 epochs, and it works really well.

From the left to the right, the first is the fundus, the second is the output of the model, the last one is the angio, groundtruth as well.
#### Global
![35-global](/utils/35-global.jpg)
#### Local
![35-local](/utils/35-local.jpg)

## How to train
### 1. Dataset preparing
We will use the paired images that being random_cropped by the official code. To do that, you should fistly download the dataset
and located them as the official website said:
- Folder structure for data-preprocessing given below. Please make sure it matches with your local repository.
```
├── Dataset
|   ├──ABNORMAL
|   ├──NORMAL
```
- Then you will get a "data" directory
```
├── data
|   ├──Images
|   ├──Masks
```
it contains a lot of paired images, they will be used later

### 2. Change the train_config.yaml
You could find it in the "config" directory, it contains some choices used in the training. 
You can config your dataset path here. We recognize you to choose the "official_data_path", since the "data_path" could have some problem.

### 3. Start training
Since every choices is defined by yaml file, we can directily start training
```python train.py```


