# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.10.7)
project(BertOpt VERSION 0.1.0)

set(BACKENDS "TF" CACHE STRING "Backends for which to build the BertOp 'TF' (tensorflow) and/or 'PT' (pytorch), default: 'TF'")

add_subdirectory(src/bentomlx/intel/custom_ops/_core)

# [WIP] by kchutkie
# add_subdirectory(python)
