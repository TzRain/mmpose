#! /bin/bash -x
CONFIG=$1
./tools/dist_train.sh $CONFIG.py 2 --work-dir ./work_dirs/$CONFIG