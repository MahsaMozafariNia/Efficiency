#!/bin/bash

module load apptainer

arch="resnet18"

# dataset="MPC"
dataset="SynthDisjoint"


# dataset="ADM"
# dataset="BigGAN"
# dataset="Midjourney"
# dataset="glide"
# dataset="stable_diffusion_v_1_4"
# dataset="VQDM"




# modifier_string="None,None,None,None,None,None"

# modifier_string="nature,nature,nature,nature,nature"
modifier_string="ai,ai,ai,ai,ai,ai"


# modifier_string="ai,nature,ai"
# modifier_string="ai,ai,nature"
# modifier_string="ai,nature,nature"
# modifier_string="nature,ai,nature"

# modifier_string="ai,nature,ai"
# modifier_string="nature,ai,ai"







# EVAL_MODIFIER="None"
# EVAL_MODIFIER="ai"
EVAL_MODIFIER="nature"


# task_num=1
# eval_tasknum=1
num_tasks=1





# RUN_ID="pretrained-final-LfBaseline"
# RUN_ID="pretrained-final-LfSteps"

# RUN_ID="pretrained-synthetic-LfBaseline"
# load_from="baseline"

RUN_ID="pretrained-synthetic-LfSteps"
load_from="steps"





### our model args
EPOCHS=300
dropout_factor=0.5
BATCH_SIZE=128
sparsity=0.65


# HSIC args

# ATTACK_TYPE="None"
ATTACK_TYPE="PGD"
# ATTACK_TYPE="AutoAttack"

# ATTACK_TYPE="gaussian_noise"
# ATTACK_TYPE="gaussian_blur"
# ATTACK_TYPE="saturate"
# ATTACK_TYPE="rotate"



# EVAL_ATTACK_TYPE="None"
EVAL_ATTACK_TYPE="PGD"
# EVAL_ATTACK_TYPE="AutoAttack"

# EVAL_ATTACK_TYPE="gaussian_noise"
# EVAL_ATTACK_TYPE="gaussian_blur"
# EVAL_ATTACK_TYPE="saturate"
# EVAL_ATTACK_TYPE="rotate"







setSorting="fixed"
lr_min=0.0001
lr_patience=20
lr=0.1
lr_factor=0.1
caper_epsilon=0.0
eval_interval=5
T=1.0
TAU=50

classRemovalAllowance=300

echo $RUN_ID













set_size=1


EpochAccMetric="loss"
sortOrder="descending"
EpochAccWindow=-1
EpochAccInterval=1


energyLayers=0

# for removal_metric in 'Caper'
for removal_metric in 'EpochAcc'
# for removal_metric in 'Random'
# for removal_metric in 'NoRemoval'
do
	
	for num_sets in 60
	do
		for task_num in 3 4 5
		do
			for trial in 1 2 3
			do
				eval_tasknum=$task_num
				# Use pretrained
				time srun apptainer run --nv ~/pytorch-advcorr2.simg \
				python "eval-steps.py" --arch=$arch --dataset=$dataset --run_id=$RUN_ID --dropout_factor=$dropout_factor \
				--prune_perc_per_layer=$sparsity --train_epochs=$EPOCHS --finetune_epochs=150 \
				--lr=$lr --lr_patience=$lr_patience --lr_factor=$lr_factor --lr_min=$lr_min \
				--tau=$TAU --attack_type=$ATTACK_TYPE  --trial_num=$trial --batch_size=$BATCH_SIZE --set_size=$set_size --num_sets=$num_sets \
				--task_num=$task_num  --removal_metric=$removal_metric --classRemovalAllowance=$classRemovalAllowance --load_from=$load_from --modifier_string=$modifier_string \
				--caper_epsilon=$caper_epsilon --eval_interval=$eval_interval --T=$T --setSorting=$setSorting \
				--pretrained --energyLayers=$energyLayers --sortOrder=$sortOrder \
				--EpochAccMetric=$EpochAccMetric --EpochAccEpochs=$EpochAccWindow --EpochAccInterval=$EpochAccInterval \
				--eval_attack_type=$EVAL_ATTACK_TYPE --eval_modifier=$EVAL_MODIFIER --eval_tasknum=$eval_tasknum


			done
		done
	done
done





echo "====================================================="
date
