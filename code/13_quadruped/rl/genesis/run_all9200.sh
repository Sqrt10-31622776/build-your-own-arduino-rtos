#!/bin/sh
# Waits for any running train.py to finish, then extends the remaining runs to ~9200 iterations, one at a time.
cd "$(dirname "$0")"
while pgrep -f "train.py" > /dev/null; do sleep 60; done
sh run_extend2.sh 8192 4600 \
  baseline:1:baseline_s1_ext:baseline_s1_ext2 \
  joint:1:joint_s1_ext:joint_s1_ext2 \
  baseline:3:baseline_s3_ext:baseline_s3_ext2 \
  joint:3:joint_s3_ext:joint_s3_ext2 \
  baseline:4:baseline_s4:baseline_s4_ext \
  joint:4:joint_s4:joint_s4_ext \
  baseline:5:baseline_s5:baseline_s5_ext \
  joint:5:joint_s5:joint_s5_ext
