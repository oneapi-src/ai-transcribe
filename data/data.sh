#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# Adopted from the source: https://github.com/opendcd/opendcd/blob/master/egs/librispeech-get-data.sh

#Fetch some test data and make s short test
if [ ! -f test-clean.tar.gz ]; then
  wget http://www.openslr.org/resources/12/test-clean.tar.gz
  tar -zxf test-clean.tar.gz
fi


if [ ! -f dev-clean.tar.gz ]; then
  wget http://www.openslr.org/resources/12/dev-clean.tar.gz
  tar -zxf dev-clean.tar.gz
fi
