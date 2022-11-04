#!/bin/bash
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

#Please give the path of precompute_pca512_cls128_mean_pooled from prepared_audio folder
TASK_DATA='<absolute path of cloned fairseq>/prepared_audio/precompute_pca512_cls128_mean_pooled'

#Intel Model - please give the same path where the model was saved earlier
MODEL_PATH='<absolute path of cloned fairseq>/model/stock/checkpoint_best.pt'

test_eval_result="<absolute path of the cloned repo>/test_eval_result_intel"
mkdir -p $test_eval_result

#python ${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/w2vu_generate.py --config-dir ${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/config/generate/ --config-name viterbi \
#OMP_NUM_THREADS=8 KMP_BLOCKTIME=0 python -m intel_extension_for_pytorch.cpu.launch --disable_iomp --enable_tcmalloc --disable_numactl ${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/w2vu_generate.py --config-dir ${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/config/generate/ --config-name viterbi \
OMP_NUM_THREADS=8 KMP_BLOCKTIME=0 python -m intel_extension_for_pytorch.cpu.launch --disable_iomp --enable_tcmalloc --disable_numactl ${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/w2vu_generate.py --config-dir ${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/config/generate/ --config-name viterbi \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/ \
fairseq.task.data=${TASK_DATA} \
fairseq.dataset.gen_subset=train \
beam=1500 \
targets=wrd \
results_path=${test_eval_result} \
fairseq.common.intel=true \
fairseq.common.inc=true \
fairseq.dataset.batch_size=1 \
fairseq.common_eval.path=${MODEL_PATH}