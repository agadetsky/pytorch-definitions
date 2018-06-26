# Conditional Generators of Words Definitions

This repo contains code for our paper [Conditional Generators of Words Definitions]().

__Abstract__

We explore recently introduced definition modeling technique that provided the tool for evaluation of different distributed
vector representations of words through modeling dictionary definitions of words. In this work, we study the problem of word ambiguities in definition modeling and propose a possible solution by employing latent variable modeling and soft attention mechanisms. Our quantitative and qualitative evaluation and analysis of the model shows that taking into account words ambi-guity and polysemy leads to performance improvement.

# Environment requirements and Data Preparation

Install conda environment with the following packages:

```
Python 3.6
Pytorch 0.4
Numpy 1.14
Tqdm 4.23
Gensim 3.4
```

To install AdaGram software to use Adaptive conditioning:
```
install Julia 0.6 from binaries https://julialang.org/downloads/
then add in ~/.bashrc
alias julia='JULIA_BINARY_PATH/bin/julia'
then in julia interpreter install following packages:
Pkg.clone("https://github.com/mirestrepo/AdaGram.jl")
Pkg.build("AdaGram")
Pkg.add("ArgParse")
Pkg.add("JSON")
Pkg.add("NPZ")
exit()
Then add in ~/.bashrc
export PATH="JULIA_BINARY_PATH/bin:$PATH"
export LD_LIBRARY_PATH="JULIA_INSTALL_PATH/v0.6/AdaGram/lib:$LD_LIBRARY_PATH"
and finally
source ~/.bashrc
```
To install Mosesdecoder (for BLEU) follow instructions on the [official site](http://www.statmt.org/moses/?n=Development.GetStarted)

To get data for language model (LM) pretraining:
```
cd pytorch-definitions
mkdir data
cd data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
```
To get data for Google word vectors use [official site](https://code.google.com/archive/p/word2vec/). You need .bin.gz file.<br/> Don't forget to `gunzip` downloaded file to extract binaries

Adaptive Skip-gram vectors are available upon request. Also you can train your owns using instructions in the [official repo](https://github.com/sbos/AdaGram.jl)

The Definition Modeling data is available upon request because of Oxford Dictionaries distribution license. Also you can collect your own. If you want to collect your own, then you should prepare 3 datasplits: train, test and val. Each datasplit is json file with the following format:

```
data = [
  [
    ["word"],
    ["word1", "word2", ...],
    ["word1", "word2", ...]
  ],
  ...
]
So i-th element of the data:
data[i][0][0] - word being defined (string)
data[i][1] - definition (list of strings)
data[i][2] - context to understand word meaning (list of strings)
```

# Usage
Firstly, you need to prepare vocabs, vectors and etc for using model:

To prepare vocabs use `python prep_vocab.py`

```
usage: prep_vocab.py [-h] --defs DEFS [DEFS ...] [--lm LM [LM ...]] [--same]
                     --save SAVE [--save_context SAVE_CONTEXT] --save_chars
                     SAVE_CHARS

Prepare vocabularies for model

optional arguments:
  -h, --help            show this help message and exit
  --defs DEFS [DEFS ...]
                        location of json file with definitions.
  --lm LM [LM ...]      location of txt file with text for LM pre-training
  --same                use same vocab for definitions and contexts
  --save SAVE           where to save prepaired vocabulary (for words from
                        definitions)
  --save_context SAVE_CONTEXT
                        where to save vocabulary (for words from contexts)
  --save_chars SAVE_CHARS
                        where to save char vocabulary (for chars from all
                        words)
```

To prepare w2v vectors use `python prep_w2v.py`
```
usage: prep_w2v.py [-h] --defs DEFS [DEFS ...] --save SAVE [SAVE ...] --w2v
                   W2V

Prepare word vectors for Input conditioning

optional arguments:
  -h, --help            show this help message and exit
  --defs DEFS [DEFS ...]
                        location of json file with definitions.
  --save SAVE [SAVE ...]
                        where to save files
  --w2v W2V             location of binary w2v file
```

To prepare Adagram vectors use `julia prep_ada.jl`
```
usage: prep_ada.jl --defs DEFS [DEFS...] --save SAVE [SAVE...]
                   --ada ADA [-h]

Prepare word vectors for Input-Adaptive conditioning

optional arguments:
  --defs DEFS [DEFS...]
                        location of json file with definitions.
  --save SAVE [SAVE...]
                        where to save files
  --ada ADA             location of AdaGram file
  -h, --help            show this help message and exit
```
If you want to init embedding matrix of the model with Google word vectors then prepare it using
`python prep_embedding_matrix.py` and then use path to saved weights as `----w2v_weights` in `train.py`
```
usage: prep_embedding_matrix.py [-h] --voc VOC --w2v W2V --save SAVE

Prepare word vectors for embedding layer in the model

optional arguments:
  -h, --help   show this help message and exit
  --voc VOC    location of model vocabulary file
  --w2v W2V    location of binary w2v file
  --save SAVE  where to save prepaired matrix
```

Now all is already ready for model usage!

To train model use `python train.py`
```
usage: train.py [-h] [--pretrain] --voc VOC [--train_defs TRAIN_DEFS]
                [--eval_defs EVAL_DEFS] [--test_defs TEST_DEFS]
                [--input_train INPUT_TRAIN] [--input_eval INPUT_EVAL]
                [--input_test INPUT_TEST]
                [--input_adaptive_train INPUT_ADAPTIVE_TRAIN]
                [--input_adaptive_eval INPUT_ADAPTIVE_EVAL]
                [--input_adaptive_test INPUT_ADAPTIVE_TEST]
                [--context_voc CONTEXT_VOC] [--ch_voc CH_VOC]
                [--train_lm TRAIN_LM] [--eval_lm EVAL_LM] [--test_lm TEST_LM]
                [--bptt BPTT] --nx NX --nlayers NLAYERS --nhid NHID
                --rnn_dropout RNN_DROPOUT [--use_seed] [--use_input]
                [--use_input_adaptive] [--use_input_attention]
                [--n_attn_embsize N_ATTN_EMBSIZE] [--n_attn_hid N_ATTN_HID]
                [--attn_dropout ATTN_DROPOUT] [--attn_sparse] [--use_ch]
                [--ch_emb_size CH_EMB_SIZE]
                [--ch_feature_maps CH_FEATURE_MAPS [CH_FEATURE_MAPS ...]]
                [--ch_kernel_sizes CH_KERNEL_SIZES [CH_KERNEL_SIZES ...]]
                [--use_hidden] [--use_hidden_adaptive]
                [--use_hidden_attention] [--use_gated] [--use_gated_adaptive]
                [--use_gated_attention] --lr LR --decay_factor DECAY_FACTOR
                --decay_patience DECAY_PATIENCE --num_epochs NUM_EPOCHS
                --batch_size BATCH_SIZE --clip CLIP --random_seed RANDOM_SEED
                --exp_dir EXP_DIR [--w2v_weights W2V_WEIGHTS]
                [--fix_embeddings] [--fix_attn_embeddings] [--lm_ckpt LM_CKPT]
                [--attn_ckpt ATTN_CKPT]

Script to train a model

optional arguments:
  -h, --help            show this help message and exit
  --pretrain            whether to pretrain model on LM dataset or train on
                        definitions
  --voc VOC             location of vocabulary file
  --train_defs TRAIN_DEFS
                        location of json file with train definitions.
  --eval_defs EVAL_DEFS
                        location of json file with eval definitions.
  --test_defs TEST_DEFS
                        location of json file with test definitions
  --input_train INPUT_TRAIN
                        location of train vectors for Input conditioning
  --input_eval INPUT_EVAL
                        location of eval vectors for Input conditioning
  --input_test INPUT_TEST
                        location of test vectors for Input conditioning
  --input_adaptive_train INPUT_ADAPTIVE_TRAIN
                        location of train vectors for InputAdaptive
                        conditioning
  --input_adaptive_eval INPUT_ADAPTIVE_EVAL
                        location of eval vectors for InputAdaptive
                        conditioning
  --input_adaptive_test INPUT_ADAPTIVE_TEST
                        location test vectors for InputAdaptive conditioning
  --context_voc CONTEXT_VOC
                        location of context vocabulary file
  --ch_voc CH_VOC       location of CH vocabulary file
  --train_lm TRAIN_LM   location of txt file train LM data
  --eval_lm EVAL_LM     location of txt file eval LM data
  --test_lm TEST_LM     location of txt file test LM data
  --bptt BPTT           sequence length for BackPropThroughTime in LM
                        pretraining
  --nx NX               size of embeddings
  --nlayers NLAYERS     number of LSTM layers
  --nhid NHID           size of hidden states
  --rnn_dropout RNN_DROPOUT
                        probability of RNN dropout
  --use_seed            whether to use Seed conditioning or not
  --use_input           whether to use Input conditioning or not
  --use_input_adaptive  whether to use InputAdaptive conditioning or not
  --use_input_attention
                        whether to use InputAttention conditioning or not
  --n_attn_embsize N_ATTN_EMBSIZE
                        size of InputAttention embeddings
  --n_attn_hid N_ATTN_HID
                        size of InputAttention linear layer
  --attn_dropout ATTN_DROPOUT
                        probability of InputAttention dropout
  --attn_sparse         whether to use sparse embeddings in InputAttention or
                        not
  --use_ch              whether to use CH conditioning or not
  --ch_emb_size CH_EMB_SIZE
                        size of embeddings in CH conditioning
  --ch_feature_maps CH_FEATURE_MAPS [CH_FEATURE_MAPS ...]
                        list of feature map sizes in CH conditioning
  --ch_kernel_sizes CH_KERNEL_SIZES [CH_KERNEL_SIZES ...]
                        list of kernel sizes in CH conditioning
  --use_hidden          whether to use Hidden conditioning or not
  --use_hidden_adaptive
                        whether to use HiddenAdaptive conditioning or not
  --use_hidden_attention
                        whether to use HiddenAttention conditioning or not
  --use_gated           whether to use Gated conditioning or not
  --use_gated_adaptive  whether to use GatedAdaptive conditioning or not
  --use_gated_attention
                        whether to use GatedAttention conditioning or not
  --lr LR               initial lr
  --decay_factor DECAY_FACTOR
                        factor to decay lr
  --decay_patience DECAY_PATIENCE
                        after number of patience epochs - decay lr
  --num_epochs NUM_EPOCHS
                        number of epochs to train
  --batch_size BATCH_SIZE
                        batch size
  --clip CLIP           value to clip norm of gradients to
  --random_seed RANDOM_SEED
                        random seed
  --exp_dir EXP_DIR     where to save all stuff about training
  --w2v_weights W2V_WEIGHTS
                        path to pretrained embeddings to init
  --fix_embeddings      whether to update embedding matrix or not
  --fix_attn_embeddings
                        whether to update attention embedding matrix or not
  --lm_ckpt LM_CKPT     path to pretrained language model weights
  --attn_ckpt ATTN_CKPT
                        path to pretrained Attention module
```

For example to train simple language model use:
```
python train.py --voc VOC_PATH --nx 300 --nhid 300 --rnn_dropout 0.5 --lr 0.001 --decay_factor 0.1 --decay_patience 0
--num_epochs 1 --batch_size 16 --clip 5 --random_seed 42 --exp_dir DIR_PATH -bptt 30
--pretrain --train_lm PATH_TO_WIKI_103_TRAIN --eval_lm PATH_TO_WIKI_103_EVAL --test_lm PATH_TO_WIKI_103_TEST
```

For example to train `Seed + Input` model use:
```
python train.py --voc VOC_PATH --nx 300 --nhid 300 --rnn_dropout 0.5 --lr 0.001 --decay_factor 0.1 --decay_patience 0
--num_epochs 1 --batch_size 16 --clip 5 --random_seed 42 --exp_dir DIR_PATH
--train_defs TRAIN_SPLIT_PATH --eval_defs EVAL_DEFS_PATH --test_defs TEST_DEFS_PATH --use_seed
--use_input --input_train PREPARED_W2V_TRAIN_VECS --input_eval PREPARED_W2V_EVAL_VECS --input_test PREPARED_W2V_TEST_VECS
```

To train `Seed + Input` model with pretraining as unconditional LM provide path to pretrained LM weights<br/>as `--lm_ckpt` argument in `train.py`

To generate using model use `python generate.py`
```
usage: generate.py [-h] --params PARAMS --ckpt CKPT --tau TAU --n N --length
                   LENGTH [--prefix PREFIX] [--wordlist WORDLIST]
                   [--w2v_binary_path W2V_BINARY_PATH]
                   [--ada_binary_path ADA_BINARY_PATH]
                   [--prep_ada_path PREP_ADA_PATH]

Script to generate using model

optional arguments:
  -h, --help            show this help message and exit
  --params PARAMS       path to saved model params
  --ckpt CKPT           path to saved model weights
  --tau TAU             temperature to use in sampling
  --n N                 number of samples to generate
  --length LENGTH       maximum length of generated samples
  --prefix PREFIX       prefix to read until generation starts
  --wordlist WORDLIST   path to word list with words and contexts
  --w2v_binary_path W2V_BINARY_PATH
                        path to binary w2v file
  --ada_binary_path ADA_BINARY_PATH
                        path to binary ada file
  --prep_ada_path PREP_ADA_PATH
                        path to prep_ada.jl script
```

To evaluate model use `python eval.py`
```
usage: eval.py [-h] --params PARAMS --ckpt CKPT --datasplit DATASPLIT --type
               TYPE [--wordlist WORDLIST] [--tau TAU] [--n N]
               [--length LENGTH]

Script to evaluate model

optional arguments:
  -h, --help            show this help message and exit
  --params PARAMS       path to saved model params
  --ckpt CKPT           path to saved model weights
  --datasplit DATASPLIT
                        train, val or test set to evaluate on
  --type TYPE           compute ppl or bleu
  --wordlist WORDLIST   word list to evaluate on (by default all data will be
                        used)
  --tau TAU             temperature to use in sampling
  --n N                 number of samples to generate
  --length LENGTH       maximum length of generated samples
```

To measure BLEU for trained model, firstly evaluate it using `--bleu` argument in `eval.py`<br/>
and then compute bleu using `python bleu.py`
```
usage: bleu.py [-h] --ref REF --hyp HYP --n N [--with_contexts] --bleu_path
               BLEU_PATH --mode MODE

Script to compute BLEU

optional arguments:
  -h, --help            show this help message and exit
  --ref REF             path to file with references
  --hyp HYP             path to file with hypotheses
  --n N                 --n argument used to generate --ref file using eval.py
  --with_contexts       whether to consider contexts or not when compute BLEU
  --bleu_path BLEU_PATH
                        path to mosesdecoder sentence-bleu binary
  --mode MODE           whether to average or take random example per word
```

Also you can pretrain Attention module using `python train_attention_skipgram.py` and<br/>
then use path to saved weights as `--attn_ckpt` argument in `train.py`
```
usage: train_attention_skipgram.py [-h] [--data DATA] --context_voc
                                   CONTEXT_VOC [--prepared] --window WINDOW
                                   --random_seed RANDOM_SEED [--sparse]
                                   --vec_dim VEC_DIM --attn_hid ATTN_HID
                                   --attn_dropout ATTN_DROPOUT --lr LR
                                   --batch_size BATCH_SIZE --num_epochs
                                   NUM_EPOCHS --exp_dir EXP_DIR

Script to train a AttentionSkipGram model

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           path to data
  --context_voc CONTEXT_VOC
                        path to context voc for DefinitionModelingModel is
                        necessary to save pretrained attention module,
                        particulary embedding matrix
  --prepared            whether to prepare data or use already prepared
  --window WINDOW       window for AttentionSkipGram model
  --random_seed RANDOM_SEED
                        random seed for training
  --sparse              whether to use sparse embeddings or not
  --vec_dim VEC_DIM     vector dim to train
  --attn_hid ATTN_HID   hidden size in attention module
  --attn_dropout ATTN_DROPOUT
                        dropout prob in attention module
  --lr LR               initial lr to use
  --batch_size BATCH_SIZE
                        batch size to use
  --num_epochs NUM_EPOCHS
                        number of epochs to train
  --exp_dir EXP_DIR     where to save weights, prepared data and logs
```
# Citation

```
@article {definitions2018,
  title  = {Conditional Generators of Words Definitions},
  author = {Gadetsky, Artyom and Yakubovskiy, Ilya and Vetrov, Dmitry},
  booktitle = {Proceedings of the 56th Annual Meeting of Association for Computational Linguistics},
  year = {2018}
}
```
