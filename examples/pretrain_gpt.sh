#!/bin/bash

# Runs the "857M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=/tmp/meg-gpt2/ckpt
VOCAB_FILE=./gpt2-vocab.json
MERGE_FILE=./gpt2-merges.txt
DATA_PATH=./meg-gpt2_text_document

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 4
	--pipeline-model-parallel-size 1 
)

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr 0.00015 \
    --train-iters 10 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

mkdir -p $CHECKPOINT_PATH

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    ${MODEL_PARALLEL_ARGS[@]} \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

rm -r $CHECKPOINT_PATH