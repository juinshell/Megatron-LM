mkdir -p /tmp/ckpt
mkdir -p /tmp/meg-gpt2/tensorboard

# on host
bash ./examples/gpt3/train_gpt3_175b_distributed.sh \
    /tmp/ckpt \
    /tmp/meg-gpt2/tensorboard \
    gpt2-vocab.json \
    gpt2-merges.txt \
    meg-gpt2_text_document

# in container
bash ./examples/gpt3/train_gpt3_175b_distributed.sh \
    /tmp/ckpt \
    /tmp/meg-gpt2/tensorboard \
    /state/partition/jxdeng/gpt2/gpt2-vocab.json \
    /state/partition/jxdeng/gpt2/gpt2-merges.txt \
    /state/partition/jxdeng/gpt2/meg-gpt2_text_document

rm -rf /tmp/ckpt/*
rm -rf /tmp/meg-gpt2/tensorboard/*

# cp gpt2-* /state/partition/jxdeng/gpt2/
# cp meg-gpt2_text_document* /state/partition/jxdeng/gpt2/