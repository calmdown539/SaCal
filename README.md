Requirements
----
This project is run in a conda virtual environment on Ubuntu 20.04 with CUDA 11.1. 
+ torch==1.10.1+cu111
+ Python==3.8.20
+ transformers==4.30.2
+ tokenizers==0.13.3
+ huggingface-hub==0.29.3

Data preparation
----
You will first need to request access for MIMIC dataset:
+ MIMIC-III v1.4 https://physionet.org/content/mimiciii/1.4v
+ MIMIC-IV v2.0 https://physionet.org/content/mimiciv/2.0/
+ MIMIC-CXR-JPG v2.0.0 https://physionet.org/content/mimic-cxr-jpg/2.0.0/
+ MIMIC-IV-NOTE v2.2 https://physionet.org/content/mimic-iv-note/2.2/

Then follow the steps in [mimic4extract](mimic4extract/README.md) to build datasets for all tasks in directory [data].

In addition, we use _biobert-base-cased-v1.2_ as the pretrained note encoder, please download files in https://huggingface.co/dmis-lab/biobert-base-cased-v1.2, and put them into the directory [mymodel/pretrained]

Model training
----
``
python my_main.py --data_path data --ehr_path data/ehr --cxr_path data/cxr --task in-hospital-mortality,length-of-stay,decompensation,phenotyping,readmission --epochs 15 --lr 0.0001 --device cuda --seed {40,42,44,46,48}
``

