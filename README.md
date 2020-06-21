# Deep-NSCLC Using InceptionV3 and custom model:

To provide the pathology department at JSS Hospitals with highly accurate model which can detect lung cancers and also predict the mutations.

Setup:
1. Conda Environment
2. python 3.6.5
3. tensorflow-gpu 1.9.0
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

## Full Training the Network
(In progress)

## Evaluating the Networks performance
(Not started)

## Post Processing
(Not started)

## Folder Structure
