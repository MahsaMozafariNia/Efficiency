
arch="resnet18"

# dataset="MPC"
dataset="SynthDisjoint"


# dataset="ADM"
# dataset="BigGAN"
# dataset="Midjourney"
# dataset="glide"
# dataset="stable_diffusion_v_1_4"
# dataset="VQDM"
# dataset="SynthDisjoint"



# modifier_string="None,None,None,None,None,None"

# modifier_string="ai,nature,nature,nature,nature,nature"
# modifier_string="ai,ai,nature,nature,nature,nature"
# modifier_string="ai,ai,ai,nature,nature,nature"
# modifier_string="ai,ai,ai,ai,nature,nature"
# modifier_string="ai,ai,ai,ai,ai,nature"



# modifier_string="nature,ai,nature"
# modifier_string="nature,nature,ai"

# modifier_string="ai,nature,ai"
# modifier_string="nature,ai,nature"







modifier_string="nature,nature,nature,nature,nature,nature"
# modifier_string="ai,ai,ai,ai,ai,ai"



# task_num=0
num_tasks=1

# RUN_ID="pretrained-final-LfBaseline"
# RUN_ID="pretrained-final-LfSteps"

# RUN_ID="pretrained-synthetic-LfBaseline"
RUN_ID="pretrained-synthetic-LfSteps"


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






setSorting="fixed"
lr_min=0.0001
lr_patience=20
lr=0.1
lr_factor=0.1
eval_interval=5
T=1.0
TAU=50


echo $RUN_ID

# steps="step3"
# load_from="baseline"


steps="allsteps"
load_from="steps"












set_size=1



EpochAccMetric="loss"
sortOrder="descending"
EpochAccWindow=-1
EpochAccInterval=1




# for removal_metric in 'Caper'
for removal_metric in 'EpochAcc'
# for removal_metric in 'Random'
do
	for num_sets in 60
	do	
		for task_num in 0
		do
			for trial in 7 8 9
			do

				# Use pretrained
				time srun apptainer run --nv ~/pytorch-advcorr2.simg \
				python "main-steps.py" --arch=$arch --dataset=$dataset --run_id=$RUN_ID --dropout_factor=$dropout_factor \
				--prune_perc_per_layer=$sparsity --train_epochs=$EPOCHS --finetune_epochs=150 \
				--lr=$lr --lr_patience=$lr_patience --lr_factor=$lr_factor --lr_min=$lr_min \
				--tau=$TAU --attack_type=$ATTACK_TYPE  --trial_num=$trial --batch_size=$BATCH_SIZE --set_size=$set_size --num_sets=$num_sets \
				--task_num=$task_num  --removal_metric=$removal_metric --load_from=$load_from --modifier_string=$modifier_string \
				--eval_interval=$eval_interval --T=$T --setSorting=$setSorting --steps=$steps \
				--pretrained --sortOrder=$sortOrder \
				--EpochAccMetric=$EpochAccMetric --EpochAccEpochs=$EpochAccWindow --EpochAccInterval=$EpochAccInterval



			done
		done
	done
done





echo "====================================================="
date
