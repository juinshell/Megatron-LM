# pip install nltk numpy parameterized pybind11 regex six tensorboard transformers 'black==21.4b0' 'isort>=5.5.4' apex 
# pip install git+https://github.com/NVIDIA/TransformerEngine.git@release_v1.4

# apex
# cd /workspace/shmem-triton/3rdparty/apex
# pip install -r requirements.txt
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

xz -d oscar-1GB.jsonl.xz

python ./tools/preprocess_data.py \
  --input ./oscar-1GB.jsonl \
  --output-prefix meg-gpt2 \
  --vocab-file ./gpt2-vocab.json \
  --tokenizer-type GPT2BPETokenizer \
  --merge-file ./gpt2-merges.txt \
  --append-eod \
  --workers 8

# mkdir -p /tmp/meg-gpt2/ckpt
# mkdir -p /tmp/meg-gpt2/tensorboard
# bash ./examples/gpt3/train_gpt3_175b_distributed.sh \
#     /tmp/meg-gpt2/ckpt \
#     /tmp/meg-gpt2/tensorboard \
#     ./gpt2-vocab.json \
#     ./gpt2-merges.txt \
#     ./meg-gpt2_text_document

# git clone --branch release_v1.4 --recursive https://github.com/NVIDIA/TransformerEngine.git