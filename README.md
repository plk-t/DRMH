# Multi-Label Hashing for Dependency Relations among Multiple Objectives (DRMH)

The official implementation of **Multi-Label Hashing for Dependency Relations among Multiple Objectives**

**Requirements**

* Linux with Python >= 3.7
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.7.0
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* CUDA 11.0


## Getting Started

## Train

The datasets files could be obtained from https://pan.baidu.com/s/10OBhx3FHf_RpK4rK_CACZQ, code:xuu7
Train on MIRFLICKR-25K, hash bit: 32bit 

Trained model will be saved in 'weight/flickr/'

```
python train.py --dataset flickr --hash_code_length 32 --num_classes 38
```



Train on IAPRTC12, hash bits: 32bit 

Trained model will be saved in 'weight/iaprtc/'

```
python train.py --dataset iaprtc --hash_code_length 32 --num_classes 22
```



Train on NUS-WIDE, hash bit: 32bit 

Trained model will be saved in 'weight/nuswide/'

```
python train.py --dataset nuswide --hash_code_length 32 --num_classes 21
```



Train on LOCKED-BIKE, hash bit: 4bit 

Trained model will be saved in 'weight/bikelock/'

```
python train.py --dataset bikelock --hash_code_length 4 --num_classes 4
```



## Test


It will take a long time to generate hash codes for database, because of the large-scale data size for database


Test for MIRFLICKR-25K, hash bit: 32bit 

```
python eval.py --dataset flickr --hash_code_length 32 --num_classes 38 --weight_pth 'your weight directory'
```



Test for IAPRTC12, hash bits: 32bit 

```
python train.py --dataset iaprtc --hash_code_length 32 --num_classes 22 --weight_pth 'your weight directory'
```



Test for NUS-WIDE, hash bit: 32bit 

```
python train.py --dataset nuswide --hash_code_length 32 --num_classes 21 --weight_pth 'your weight directory'
```



Test for  LOCKED-BIKE, hash bit: 4bit 

```
python train.py --dataset bikelock --hash_code_length 4 --num_classes 4 --weight_pth 'your weight directory'
```

