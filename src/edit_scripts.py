# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import os

#Edit w2vu.yaml to enable and setup training with Intel® Extension for PyTorch
w2vu_path = os.path.join(os.environ['FAIRSEQ_ROOT'], "examples/wav2vec/unsupervised/config/gan/w2vu.yaml")

with open(w2vu_path, "r") as file:
    lines = file.readlines()

correct_intel = "  intel: true" + '\n'
correct_workers = "  num_workers: 0" + '\n'

enable_intel = lines[12].replace(lines[12], correct_intel)
num_workers = lines[36].replace(lines[36], correct_workers)

lines[12] = enable_intel
lines[36] = num_workers

with open(w2vu_path, "w") as file:
    file.writelines(lines)

#Edit w2vu_generate.py file to enable a successfull inference process
w2vu_generate_path = os.path.join(os.environ['FAIRSEQ_ROOT'], "examples/wav2vec/unsupervised/w2vu_generate.py")

with open(w2vu_generate_path, "r") as file:
    lines = file.readlines()

correct_line = '                logger.info(f"\\n\\nWER DROP POST QUANTIZATION : {wer_int-wer_fp}")' + '\n'

modified_line = lines[686].replace(lines[686], correct_line)

lines[686] = modified_line

with open(w2vu_generate_path, "w") as file:
    file.writelines(lines)

#Edit Inference_Intel.sh file for a correct inference setup
inference_intel = os.path.join(os.environ['SRC_DIR'], "Inference_Intel.sh")

with open(inference_intel, "r") as file:
    lines = file.readlines()

test_set = "fairseq.dataset.gen_subset=test \\" + '\n'

test_set = lines[18].replace(lines[18], test_set)

lines[18] = test_set

with open(inference_intel, "w") as file:
    file.writelines(lines)
