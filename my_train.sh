bash ./examples/gpt3/train_gpt3_175b_distributed.sh \
    /tmp/ckpt \
    /tmp/meg-gpt2/tensorboard \
    gpt2-vocab.json \
    gpt2-merges.txt \
    meg-gpt2_text_document

rm /tmp/ckpt/*
rm -rf /tmp/meg-gpt2/tensorboard/*