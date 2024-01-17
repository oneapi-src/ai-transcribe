#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

#Please give the path of precompute_pca512_cls128_mean_pooled from prepared_audio folder
TASK_DATA="${FAIRSEQ_ROOT}/prepared_audio/precompute_pca512_cls128_mean_pooled"

#Intel Model - please give the same path where the model was saved earlier
MODEL_PATH="${OUTPUT_DIR}/model/checkpoint_last.pt"

test_eval_result="${OUTPUT_DIR}/test_eval_result"
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
