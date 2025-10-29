### IRENE ###
CUDA_VISIBLE_DEVICES=0,1 python -W ignore irene.py --CLS 8 --BSZ 64 --DATA_DIR ./data --SET_TYPE test.pkl

### MedFuse ###
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python run_bl.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode train \
--epochs 10 --batch_size 32 \
--num_classes 1 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type lstm \
--save_dir medfuse_model

### MultiModN ###
python mimic.py --data_path data --ehr_path data/ehr --cxr_path data/cxr \
--device cuda --epochs 10\ 

### MulTEHR ###
python main_multehr.py --data_path data --ehr_path data/ehr --cxr_path data/cxr \
--task in-hospital-mortality,length-of-stay,decompensation,phenotyping,readmission \
--epochs 10 --lr 0.0001 --device cuda 


### FlexCare ###
python main.py --data_path data --ehr_path data/ehr --cxr_path data/cxr \
--task in-hospital-mortality,length-of-stay,decompensation,phenotyping,readmission --epochs 10 --lr 0.0005 --device cuda 


### SaCal ###
python my_main.py --data_path data --ehr_path data/ehr --cxr_path data/cxr --task in-hospital-mortality,length-of-stay,decompensation,phenotyping,readmission --epochs 10 --lr 0.0001 --device cuda --seed 40

### Design choices of SaCal, e.g., MoE-Fuser, TF-Fuser, MoE, MMoE ###
python main_dc.py --data_path data --ehr_path data/ehr --cxr_path data/cxr --task in-hospital-mortality,length-of-stay,decompensation,phenotyping,readmission --epochs 10 --lr 0.0001 --device cuda --seed 40
