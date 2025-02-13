#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#$BATCH --features=gpu_mem_80

echo "------------"
echo $SLURM_JOB_ID
echo $SLURM_NODELIST
echo "------------"

EXP_NAME=$1
OUT_DIR=$2
BATCH_SIZE=$3
EPOCHS=$4
DO_EXPERIMENT=$5
ROWS=$6
LEARNING_RATE=$7
DROPOUT_RATE=$8
LAYERS=$9
OPTIMIZER=${10}


echo "Running $EXP_NAME"

# Activate the conda environment
source /Users/mukulsherekar/yes/envs/mamai/bin/activate

EXE="/Users/mukulsherekar/pythonProject/MammographyAI/main.py"
echo "$EXE"

TRAIN_CSV_PATH="/Users/mukulsherekar/pythonProject/MammographyAI/DATA/train.csv"
VALID_CSV_PATH="/Users/mukulsherekar/pythonProject/MammographyAI/DATA/val.csv"

IMG_DIR="/Users/mukulsherekar/pythonProject/MammographyAI/DATA/" # doesnt matter but has to be there

# Run the Python  with arguments
python $EXE     --exp_name $EXP_NAME \
                --dataset_partition $DO_EXPERIMENT \
                --rows_experiment $ROWS\
                --train_csv_pth $TRAIN_CSV_PATH \
                --valid_csv_pth $VALID_CSV_PATH \
                --save_img_root_dir_pth $IMG_DIR \
                --remove_processed_pngs False \
                --data EMBED \
                --out_dir $OUT_DIR \
                --fine_tuning partial \
                --upto_freeze $LAYERS \
                --batch_size $BATCH_SIZE \
                --threads 2 \
                --num_epochs $EPOCHS\
                --optimizer $OPTIMIZER \
                --decay_every_N_epoch 5 \
                --decay_multiplier 0.95 \
                --save_every_N_epochs 1\
                --bsave_valid_results_at_epochs False \
                --random_state 42\
                --start_learning_rate $LEARNING_RATE \
                --dropout_rate $DROPOUT_RATE \
                --learned_loss_attnuation True \
                --training_augment True \
                --t_number 10 


echo "Job completed."
