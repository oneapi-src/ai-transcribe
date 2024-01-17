#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

start=$(date +"%r")
echo "start time >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: $start"


now=$(date +"%T")
echo "Current start time : $now"


PREFIX="w2v_unsup_gan_xp"


# Path to prepared_audio/precompute_pca512_cls128_mean_pooled - Prepared Audio
TASK_DATA="${FAIRSEQ_ROOT}/prepared_audio/precompute_pca512_cls128_mean_pooled"
# path to fairseq-preprocessed GAN data (phones dir) - Prepared Text data
TEXT_DATA="${FAIRSEQ_ROOT}/prepared_text/phones"

# KenLM 4-gram phoneme language model (LM data = GAN data here)
KENLM_PATH="${FAIRSEQ_ROOT}/prepared_text/phones/lm.phones.filtered.04.bin"

# Path where the model would get saved
OUT_PATH="${OUTPUT_DIR}/model"
echo $OUT_PATH
mkdir -p $OUT_PATH
# Path to the the config file of the unsupervised gan model
CONFIG_PATH="${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/config/gan"


PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX CUDA_LAUNCH_BLOCKING=1 fairseq-hydra-train \
	-m --config-dir ${CONFIG_PATH} \
	--config-name w2vu \
	+task.data=${TASK_DATA} \
	+task.text_data=${TEXT_DATA} \
	+task.kenlm_path=${KENLM_PATH} \
	checkpoint.no_epoch_checkpoints=true \
	checkpoint.keep_last_epochs=20 \
	checkpoint.save_dir=${OUT_PATH} \
	common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
	common.intel=true \
	model.code_penalty=2,4 model.gradient_penalty=1.5,2.0 \
	model.smoothness_weight=0.5,0.75,1.0 'common.seed=range(0,5)'


now1=$(date +"%T")
echo "Current end  time................................... : $now1"


difference=$(( $(date -d "$now1" "+%s") - $(date -d "$now" "+%s") ))

echo $difference


end=$(date +"%r")

echo "end time >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: $end"



