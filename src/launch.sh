#!/bin/bash
DO_EXPERIMENT='False'
ROWS='1000'
BATCH_SIZE=32
EPOCHS=5
LEARNING_RATE=1e-6
LAYERS=24
OPTIMIZER=adamW
DROPOUT_RATE=0.0
DATE=$(date +%Y-%m-%d)
JN="weighted"

EXP_NAME="${JN}_${EPOCHS}_${LEARNING_RATE}_${LAYERS}"

# Define output directory based on the job name # check and change
OUT_DIR="/Users/mukulsherekar/pythonProject/MammographyAI/OUT/2024-11-26/aug_cancer+ve/${EXP_NAME}/"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Submit the job with sbatch # make sure .sh file matches
sbatch --output="${OUT_DIR}/${EXP_NAME}.out" train.sh "$EXP_NAME" "$OUT_DIR" $BATCH_SIZE $EPOCHS $DO_EXPERIMENT $ROWS $LEARNING_RATE $DROPOUT_RATE $LAYERS $OPTIMIZER 

