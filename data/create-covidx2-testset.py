import numpy as np 
import pandas as pd
import shutil
import glob
from pathlib import Path

np.random.seed(42)

data_dir = 'test'
target_dir = 'covidx2_test'

tst = pd.read_csv('test_COVIDx2.txt', sep=' ', index_col=0)
tst = tst.drop_duplicates(subset='filename')

# create trainset subfolders if needed.
Path(target_dir + '/COVID-19').mkdir(parents=True, exist_ok=True)
Path(target_dir + '/normal').mkdir(parents=True, exist_ok=True)
Path(target_dir + '/pneumonia').mkdir(parents=True, exist_ok=True)

# trn & val
# data stored in root/subset/class/filename format.
for i in range(tst.shape[0]):
    file = Path(data_dir + '/' + tst.filename.iloc[i])
    shutil.copyfile(file, target_dir + '/' + tst.label.iloc[i] + '/' + tst.filename.iloc[i])
