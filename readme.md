# RCCNet: An Efficient Convolutional Neural Network for Histological Routine Colon Cancer Nuclei Classification
RCCNet is a CNN model which is responsible for classifying the routine colon cancer cells.

## Abstract:
Efficient and precise classification of histological cell nuclei is of utmost importance due to its potential applications
in the field of medical image analysis. It would facilitate the medical practitioners to better understand and explore various
factors for cancer treatment. The classification of histological cell nuclei is a challenging task due to the cellular
heterogeneity. This paper proposes an efficient Convolutional Neural Network (CNN) based architecture for classification of
histological routine colon cancer nuclei named as RCCNet. The main objective of this network is to keep the CNN model as simple
as possible. The proposed RCCNet model consists of only 1,512,868 learnable parameters which are significantly less compared
to the popular CNN models such as AlexNet, CIFAR-VGG, GoogLeNet, and WRN. The experiments are conducted over publicly available
routine colon cancer histological dataset "CRCHistoPhenotypes". The results of the proposed RCCNet model are compared with five
state-of-the-art CNN models in terms of the accuracy, weighted average F1 score and training time. The proposed method has
achieved a classification accuracy of 80.61% and 0.7887 weighted average F1 score. The proposed RCCNet is more efficient and
generalized terms of the training time and data over-fitting, respectively.

The overview of RCCNet is shown in the below figure.
![alt text](https://github.com/shabbeersh/RCCNet/blob/master/RCCNet.png)

For more details, please read our [paper](https://arxiv.org/abs/1810.02797).

## Requirements
Keras >= 2.1.2 <br/>
Tensorflow-gpu >= 1.2

Dataset:
In this research, we have used a publicly available routine colon cancer dataset. The dataset is available at https://warwick.ac.uk/fac/sci/dcs/research/tia/data/crchistolabelednucleihe. 

To extract the cells of dimensions 32x32x3, please use the matlab code with file name ColonCancerPatchExtraction.m

## Acknowledgements
The few blocks of code are taken from [here](https://github.com/geifmany/cifar-vgg).

## Citations
@inproceedings{basha2018rccnet,
  title={Rccnet: An efficient convolutional neural network for histological routine colon cancer nuclei classification},
  author={Basha, SH Shabbeer and Ghosh, Soumen and Babu, Kancharagunta Kishan and Dubey, Shiv Ram and Pulabaigari, Viswanath and Mukherjee, Snehasis},
  booktitle={2018 15th International Conference on Control, Automation, Robotics and Vision (ICARCV)},
  pages={1222--1227},
  year={2018},
  organization={IEEE}
}
