fastdepth
============================

This repo offers trained models and evaluation code for the [FastDepth](http://fastdepth.mit.edu/) project at MIT.
This code is borrowed from [fastdepth](https://github.com/dwofk/fast-depth) and [GuidedDecoding](https://github.com/mic-rud/GuidedDecoding), which is fastdepth unofficial training test code.
The code implements both the MobileNet encoder and the MobileNetv2 encoder, as well as the calculation flops script and the corresponding torch conversion onnx script

<p align="center">
	<img src="img/visualization.png" alt="photo not available" width="50%" height="50%">
</p>

## Contents
0. [Requirements](#requirements)
0. [Trained Models](#trained-models)
0. [Evaluation](#evaluation)
0. [Deployment](#deployment)
0. [Results](#results)
0. [Citation](#citation)

## Requirements
- Install [PyTorch](https://pytorch.org/) on a machine with a CUDA GPU. Our code was developed on a system running PyTorch v0.4.1.
- Install the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format libraries. Files in our pre-processed datasets are in HDF5 format.
  ```bash
  sudo apt-get update
  sudo apt-get install -y libhdf5-serial-dev hdf5-tools
  pip3 install h5py matplotlib imageio scikit-image opencv-python
  ```
- Download the preprocessed [NYU Depth V2](https://tinyurl.com/nyu-data-zip). 
 You don't need to extract the dataset since the code loads the entire zip file into memory when training.The NYU dataset requires 4.4G of storage space.

### Pretrained MobileNet or MobileNetV2 ###

The model file for the pretrained MobileNet used in our model definition is /pretrained/model_best.pth.tar, and MobileNetV2 used in our model definition is /pretrained/mobilenetv2_1.0-0c6065bc.pth

### Trainning

```console
python main.py --train --resolution RESOLUTION --data nyu_reduced --data_path DATASET_PATH
```
 
if you want to change the encoder,you should change the code on line 58 of the main.py 
For example:
```console
model = models.MobileNetV2SkipConcat(output_size=(224,224))
```
you can change 
```console
model = models.MobileNetSkipAdd(output_size=(224,224))
```
or
```console
model = models.MobileNetV2SkipAdd(output_size=(224,224))
```

## Evaluation ##

This step requires a valid PyTorch installation and a saved copy of the NYU Depth v2 dataset. It is meant to be performed on a host machine with a CUDA GPU, not on an embedded platform. Deployment on an embedded device is discussed in the [next section](#deployment).

you may need to download NYU test data in [here](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_test.zip)，and you need to unzip the test set.

To evaluate a model, navigate to the repo directory and run:

```console
python main.py --evaluate --test_path TESTDATA_PATH --weights_path MODEL_PRETRAINED_PATH
```

The evaluation code will report model accuracy in terms of the delta1 metric as well as RMSE in millimeters.

Note: This evaluation code was sourced and modified from [here](https://github.com/mic-rud/GuidedDecoding).


## Citation
If you reference our work, please consider citing the following:

	@inproceedings{icra_2019_fastdepth,
		author      = {{Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne}},
		title       = {{FastDepth: Fast Monocular Depth Estimation on Embedded Systems}},
		booktitle   = {{IEEE International Conference on Robotics and Automation (ICRA)}},
		year        = {{2019}}
	}
