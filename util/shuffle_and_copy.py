import os
import argparse
import glob
#import random
import numpy as np
import shutil

files_to_copy='*.png'
train_dir='train'
valid_dir='valid'
test_dir='test'

seed=4

file_list = np.array([f for f in glob.glob(files_to_copy)])

train_split=.7
valid_split=.2
test_split=1-(train_split+valid_split)

np.random.seed(seed)
np.random.shuffle(file_list)

train_cnt=int(len(file_list) * train_split)
valid_cnt=int(len(file_list) * valid_split)
test_cnt=int(len(file_list) * test_split)

train_files=file_list[:train_cnt]
valid_files=file_list[train_cnt:train_cnt+valid_cnt]
test_files=file_list[train_cnt+valid_cnt:]

print((train_cnt), (valid_cnt), (test_cnt))
print(len(train_files), len(valid_files), len(test_files))

if not os.path.exists(train_dir):
   os.mkdir(train_dir)

if not os.path.exists(valid_dir):
   os.mkdir(valid_dir)

if not os.path.exists(test_dir):
   os.mkdir(test_dir)

for f in train_files:
   shutil.copy(f, train_dir)

for f in valid_files:
   shutil.copy(f, valid_dir)

for f in test_files:
   shutil.copy(f, test_dir)




