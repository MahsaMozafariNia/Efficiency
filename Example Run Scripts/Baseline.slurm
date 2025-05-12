#!/bin/bash

module load apptainer

arch="resnet18"
# arch="vgg16"

# dataset="MPC"

# modifier_string="None,None,None,None,None,None"


dataset="SynthDisjoint"


# dataset="ADM"
# dataset="BigGAN"
# dataset="Midjourney"
# dataset="glide"
# dataset="stable_diffusion_v_1_4"
# dataset="VQDM"




modifier_string="nature,nature,nature,nature,nature,nature"
# modifier_string="ai,ai,ai,ai,ai,ai"



# modifier_string="ai,nature,nature,nature,nature,nature"
# modifier_string="ai,ai,nature,nature,nature,nature"
# modifier_string="ai,ai,ai,nature,nature,nature"
# modifier_string="ai,ai,ai,ai,nature,nature"
# modifier_string="ai,ai,ai,ai,ai,nature"

# modifier_string="nature,nature,ai"
# modifier_string="nature,ai,nature"
# modifier_string="ai,nature,ai"
# modifier_string="nature,ai,ai"



# RUN_ID="pretrained-final-LfBaseline"

RUN_ID="pretrained-synthetic-LfBaseline"




### our model args
EPOCHS=300
dropout_factor=0.5
BATCH_SIZE=128
sparsity=0.65

### learning_rate
lr_min=0.0001
lr_patience=20

# HSIC args
TAU=50
# ATTACK_TYPE="None"
ATTACK_TYPE="PGD"
# ATTACK_TYPE="AutoAttack"

# ATTACK_TYPE="gaussian_noise"
# ATTACK_TYPE="gaussian_blur"
# ATTACK_TYPE="saturate"
# ATTACK_TYPE="rotate"


T=1.0
lr=0.1
lr_factor=0.1

steps="allsteps"
load_from="baseline"
# load_from="steps"
removal_metric="NoRemoval"
eval_interval=5

echo $RUN_ID





for trial in 1 2 3
do
	for task_num in 0 1 2 3 4 5
	do

		time srun apptainer run --nv ~/pytorch-advcorr2.simg \
		python "main-steps.py" --arch=$arch --dataset=$dataset --run_id=$RUN_ID --dropout_factor=$dropout_factor \
		--prune_perc_per_layer=$sparsity --train_epochs=$EPOCHS --finetune_epochs=150 \
		--lr=$lr --lr_patience=$lr_patience --lr_factor=$lr_factor --lr_min=$lr_min \
		--tau=$TAU --attack_type=$ATTACK_TYPE  --trial_num=$trial --batch_size=$BATCH_SIZE \
		--task_num=$task_num  --removal_metric=$removal_metric --load_from=$load_from --modifier_string=$modifier_string \
		--eval_interval=$eval_interval --T=$T  --steps=$steps --pretrained




	done
done







echo "====================================================="
date
