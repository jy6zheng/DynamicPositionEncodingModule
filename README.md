## Dynamic Position Encoding

### Modified Files
* fairseq/models/transformer.py is modified to include the dynamic position encoding module in the encoder
* fairseq/criterions/label_smoothed_cross_entropy_with_alignment.py includes the criterion and loss function for comparing predicted position probabilities with the target
* data/language_pair_dataset.py and sequence_generator.py have small modifications to accomadate for the dynamic position encoding module

## Commands
Steps for processing dataset for dynamic position encoding:
* With dataset in folder wmt14_en_de (or your language pair), use following commands to install fast_align

```bash
git clone git@github.com:clab/fast_align.git
pushd fast_align
mkdir build
cd build
cmake ..
make
popd
ALIGN=fast_align/build/fast_align
TEXT=wmt14_en_de
paste $TEXT/train.en $TEXT/train.de | awk -F '\t' '{print $1 " ||| " $2}' > $TEXT/train.en-de
$ALIGN -i $TEXT/train.en-de -d -o -v > $TEXT/train.align
```
* Obtain adjusted alignments using adjust_alignments.py and changing parameters (out_file, src_file- source language dataset file, align_file- file with alignments from fast_align) within the file
* Generate binarized dataset using fairseq-preprocess

```bash
fairseq-preprocess \
--source-lang en --target-lang de \
--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
--align-suffix align \
--destdir data-bin/$TEXT --thresholdtgt 0 --thresholdsrc 0 \
--joined-dictionary \
--workers 20
```

For training a model, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train data-bin/wmt14_en_de \
--arch transformer_wmt_en_de  --share-all-embeddings \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0  \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
--warmup-updates 8000 --lr 0.0007 --stop-min-lr 1e-09 \
--criterion label_smoothed_mse_with_alignment --load-alignments \
--alignment-lambda 0.5 --label-smoothing 0.1 --weight-decay 0.0 \    
--max-tokens  4096  --save-dir ./checkpoints/wmt14_en_de \
--max-update 288000  --seed 3435 --distributed-world-size 8 \
--eval-bleu  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'  \
--eval-bleu-detok moses     --eval-bleu-remove-bpe \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--eval-bleu-print-samples --ddp-backend=no_c10d
```
And average the last five checkpoints

```bash
python scripts/average_checkpoints.py --inputs examples/dynamic-position/checkpoints/wmt14_en_de \
--num-update-checkpoints 5 --output examples/dynamic-position/checkpoints/wmt14_en_de/average.pt
```
And generate results
```bash
CUDA_VISIBLE_DEVICES=0 fairseq-generate  data-bin/wmt14_en_de \
--path checkpoints/wmt14_en_de/average.pt \
--beam 5 --batch-size 128 --remove-bpe --lenpen 0.6
```
## Additional Details
This code was developed on top of the fairseq toolkit. As a result, you need to follow the fairseq installation instructions:

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.0)
# pip install fairseq==0.10.0
```

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
