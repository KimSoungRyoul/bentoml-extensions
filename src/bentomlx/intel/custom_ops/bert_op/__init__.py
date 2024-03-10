# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import transformers

if transformers.is_torch_available():
    from . import torch

    print("inner torch is imported")
