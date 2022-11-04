# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change -start ###
inference_data_name="test_data"
wav_path='/home/azureuser/data/LibriSpeech/test-clean'
echo $wav_path
prep_scripts="prep_scripts"
destination_path=$dir'/'${inference_data_name}
### Values to change end ###

#finetuning_dict=dict.ltr.txt'
txt_path=${wav_path}
analysis_scripts="analysis"

mkdir -p ${destination_path}

python ${prep_scripts}/manifest.py ${wav_path} --dest ${destination_path} --ext flac --train-name train --valid-percent 0 --jobs -1
echo "Manifest Creation Done"

python ${prep_scripts}/labels.py --jobs 1 --tsv ${destination_path}/train.tsv --output-dir ${destination_path} --output-name train --txt-dir ${txt_path}
echo "Word file generated"

python ${prep_scripts}/dict_and_lexicon_maker.py --wrd ${destination_path}/train.wrd --lexicon ${destination_path}/lexicon.lst --dict ${destination_path}/dict.ltr.txt
echo "Dict file generated"


