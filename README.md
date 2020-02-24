# Deep-NSCLC Using InceptionV3 model:

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

System in use:


The entire project is divided into 5 phases as follows:

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

## Full Training the Network
(In progress)

## Evaluating the Networks performance
(Not started)

## Post Processing
(Not started)

## Folder Structure
