### IRENE ###
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore irene.py --CLS 8 --BSZ 64 --DATA_DIR ./data --SET_TYPE test.pkl

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
--save_dir save

### MultiModN ###
