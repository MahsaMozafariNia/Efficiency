#!/bin/bash

arch="resnet18"


# dataset="MPCoffset"
# dataset="MPC"
# dataset="KEF"
# dataset="splitcifar"
# dataset="SynthJoint"
dataset="SynthDisjoint"


# modifier="None"
# modifier="CIFAR100Full"
# modifier="OnlyCIFAR100"
# modifier="ai"
modifier="nature"


task_num=5



# HSIC args

# ATTACK_TYPE="gaussian_noise"
# ATTACK_TYPE="rotate"
# ATTACK_TYPE="gaussian_blur"
# ATTACK_TYPE="saturate"



echo $RUN_ID










###############################################################################################################################
### Using scheduler

for ATTACK_TYPE in "gaussian_noise" "rotate" "gaussian_blur" "saturate"
do
	# Use pretrained
	time srun apptainer run --nv ~/pytorch-advcorr2.simg \
	python "corruptDatasets.py" --arch=$arch --dataset=$dataset \
	--attack_type=$ATTACK_TYPE --task_num=$task_num --dataset_modifier=$modifier
done



echo "====================================================="
date
