#! /bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$5 #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 1 \
       --num-layers 12 \
       --hidden-size 512 \
       --num-attention-heads 8 \
       --micro-batch-size 1 \
       --global-batch-size 1 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 5000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file data/gpt2-vocab.json \
       --merge-file data/gpt2-merges.txt \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 10 \
       --save-interval 500 \
       --eval-interval 100 \
       --eval-iters 10 \
       --fp16