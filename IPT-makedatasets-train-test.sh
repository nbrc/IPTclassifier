#! /bin/sh

SCRIPT_RELATIVE_DIR=$(dirname "${BASH_SOURCE[0]}")
cd $SCRIPT_RELATIVE_DIR
pwd

echo "LAUNCH MAKE TRAIN TEST CODE"
echo "CREATE DATASETS"
python3 IPT-makedatasets-train-test.py
