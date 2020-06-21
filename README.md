# Explainable Deep-NSCLC Model:

To provide the pathology department at JSS Hospitals with highly accurate model which can detect lung cancers and also predict the mutations.

Setup:
1. Conda Environment
2. python 3.6.5
3. Pytorch 
4. numpy 1.14.3
5. matplotlib 2.1.2
6. sklearn
7. scipy 1.1.0
8. openslide-python 1.1.1
9. Pillow 5.1.0
10. libiconv-1.15 

### System in use:

| Description  | Value |
| ------------- | ------------- |
| 2x Nvidia Quadro P5000   | 32GB|  
| 1x Nvidia Quadro P400   | 2GB |
| Architecture:      | x86_64        |
| CPU op-mode(s):   | 32-bit, 64-bit| 
|CPU(s):      | 72  |
| On-line CPU(s) list:    | 0-71|  
| Thread(s) per core:    |2 |
| Core(s) per socket:   | 18|  
| Vendor ID:    | GenuineIntel| |
| CPU family:   | 6|  
| Model:      | 85  |
| Model name:    |  Intel(R) Xeon(R) Gold 6139 CPU @ 2.30GHz|  
| CPU MHz:    | 2259.722 |
| CPU max MHz:  |   2301.0000|  
| BogoMIPS:    | 4601.79  |
| NUMA node0 CPU(s):   | 0-17,36-53|  
| NUMA node1 CPU(s):   | 18-35,54-71 |

<br />
The entire project is divided into 5 phases as follows:<br /> 

## Data Acquisition:
Collecting whole slide images for Adenocarcinoma(LCAD) and Squamous Cell Carcinoma(LUSC) from TCGA using GDC Tool.

- TCGA - https://portal.gdc.cancer.gov/
- GDC Tool for extracting large dataset:  https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

DataSet division:
a) 504 .svs images of LUAD
b) 512 .svs images of LUSC
c) 502 .svs images of Normal Cells

Addition data for testing the model is be given by JSS Hospitals with x40, x20 maginfication (stained).

### Files required:
- gdc-manifest.txt
- gdc-metadata.json
- uuid.txt

## Pre Processing
Convert all downloaded .svs files to .jpg image format. 
Use the code availabe in "convert-svs-to-jpg/convert.py"
See folder convert-svs-to-jpg/
<br />
Users can specify the number of cores to convert very large svs images faster.

## Networks:
Inception V3 architecture:<br>
<img width="628" src="https://github.com/santos97/deep-NSCLC/blob/master/images/68747470733a2f2f7777772e50657465724d6f7373416d6c416c6c52657365617263682e636f6d2f6d656469612f696d616765732f7265706f7369746f726965732f434e4e2e6a7067.jpeg">
<br>
Custom Network:
<br>
<img src="https://github.com/santos97/deep-NSCLC/blob/master/images/custom_archi.png" width="628">

## Full Training the Network
Training V3:
<br>
<img src="https://github.com/santos97/deep-NSCLC/blob/master/images/v3_train.png" width="628">
<br>
Training Custom Net:
<br>
<img src="https://github.com/santos97/deep-NSCLC/blob/master/images/custom_training.png" width="628">

## Networks performance:
_Inception V3:_ <br>
Key findings:
<br>
1.A scheduler to dynamically change learning rate based on validation loss.<br>
2.Weight decay method to overcome very early overfitting.<br>
3.A dropout node (probability ~ 0.8) was used to avoid overfitting.<br>
Obtained accuracy ~84%<br>
<br>

_Custom Net:_ <br>
Key findings:
<br>
1.The model learns well on train set and performs well on validation set.<br>
2.Even though the model fluctuates on validation set, at epoch 19 we get the highest accuracy and lowest loss.<br>
3.Only normalization and dropout techniques were used to prevent overfitting.<br>
4.The model performs well on our small dataset.<br>
Obtained accuracy ~98%



## Explainable Net:
GradCAM Technique: Explainability highlights activated features of a input object in a neural network during a forward pass.
These explainability can be used to understand which regions of the input caused the model to predict a particular output.
<img src="https://github.com/santos97/deep-NSCLC/blob/master/images/explainable.png" width="628">

## Results:
<img src="https://github.com/santos97/deep-NSCLC/blob/master/images/r1.png" width="628">
Correct prediction of LUAD class by the model with 95% accuracy.The yellow and red regions are regions where our model identifies region of high malignancy.
These explanations by our model helps pathologists pinpoint the area of malignancy.

<br>
<img src="https://github.com/santos97/deep-NSCLC/blob/master/images/r2.png" width="628">
