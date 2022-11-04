#!/bin/bash
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

#FULL DATA for batch inference
TASK_DATA='<absolute path of cloned fairseq>/prepared_audio/precompute_pca512_cls128_mean_pooled'

#Stock Model - please give the same path where the model was saved ealrier
MODEL_PATH='<absolute path of cloned fairseq>/model/stock/checkpoint_best.pt'

test_eval_result="<absolute path of clonned repo>/test_eval_result_stock"
mkdir -p $test_eval_result

python ${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/w2vu_generate.py --config-dir ${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/config/generate/ --config-name viterbi \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/ \
fairseq.task.data=${TASK_DATA} \
fairseq.dataset.gen_subset=train \
beam=1500 \
targets=wrd \
results_path=${test_eval_result} \
fairseq.common.intel=false \
fairseq.common.inc=true \
fairseq.dataset.batch_size=1000 \
fairseq.common_eval.path=${MODEL_PATH}

