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

System in use:


The entire project is divided into 5 phases as follows:

## Data Acquisition:
Collecting whole slide images for Adenocarcinoma(LCAD) and Squamous Cell Carcinoma(LUSC) from TCGA using GDC Tool.

TCGA - ttps://portal.gdc.cancer.gov/
GDC Tool for extracting large dataset:  https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

### Files required:
- Manifest.txt
- gdc.json
- uuid.txt

## Pre Processing
(In progress)

## Full Training the Network
(In progress)

## Evaluating the Networks performance
(Not started)

## Post Processing
(Not started)