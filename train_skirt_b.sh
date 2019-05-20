#!/usr/bin/env bash
set -ex

NICKNAME=zalando_mg_skirt_B

TRAIN_LOG_FILENAME="train_`date +%Y%m%d_%H%M%S`_"${NICKNAME}.log
VAL_LOG_FILENAME="val_`date +%Y%m%d_%H%M%S`_"${NICKNAME}.log


python train_skirt_b.py 2>&1 | tee ${TRAIN_LOG_FILENAME}

echo "Train... Done."