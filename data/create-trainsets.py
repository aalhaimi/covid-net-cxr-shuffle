import numpy as np 
import pandas as pd
import shutil
import glob
from pathlib import Path

np.random.seed(42)

data_dir = 'train'
target_dir = 'trainset'
subfolders = ['trn', 'val']
target_dir_balanced = 'trainset_balanced' 
subfolders_balanced = ['trnb', 'valb']

data = pd.read_csv('train_split_v3.txt', sep=' ', index_col=0)
data = data.drop_duplicates(subset='filename')
data = data.sample(frac=1, random_state=42)

# trn/val split
trn_size = int(round(data.shape[0]*0.7)) 
trn = data[:trn_size]
val = data[trn_size:]

# create trainset subfolders if needed.
for i in subfolders:
    Path(target_dir + '/' + i + '/COVID-19').mkdir(parents=True, exist_ok=True)
    Path(target_dir + '/' + i + '/normal').mkdir(parents=True, exist_ok=True)
    Path(target_dir + '/' + i + '/pneumonia').mkdir(parents=True, exist_ok=True)

# trn & val
# data stored in root/subset/class/filename format.
for i in range(trn.shape[0]):
    file = Path(data_dir + '/' + trn.filename.iloc[i])
    shutil.copyfile(file, target_dir + '/trn/' + trn.label.iloc[i] + '/' + trn.filename.iloc[i])
for i in range(val.shape[0]):
    file = Path(data_dir + '/' + val.filename.iloc[i])
    shutil.copyfile(file, target_dir + '/val/' + val.label.iloc[i] + '/' + val.filename.iloc[i])


#--------  Optional -------- #
# comment out this section if you want to balance the classes differenly.
# if this is used, the dataset can pass directly through an augmentation method. 

for sf in subfolders_balanced: 
    normals = []
    covid19 = []
    pneumonia = []
    normals_b = []
    covid19_b = []
    pneumonia_b = []
    Path(target_dir_balanced).mkdir(parents=True, exist_ok=True)
    Path(target_dir_balanced + '/' + sf +'/COVID-19').mkdir(parents=True, exist_ok=True)
    Path(target_dir_balanced + '/' + sf + '/normal').mkdir(parents=True, exist_ok=True)
    Path(target_dir_balanced + '/' + sf + '/pneumonia').mkdir(parents=True, exist_ok=True)

    for filepath in glob.iglob(target_dir + '/' + sf[:-1] + '/normal/*'):
        normals.append(filepath)
    for filepath in glob.iglob(target_dir + '/' + sf[:-1] + '/COVID-19/*'):
        covid19.append(filepath)
    for filepath in glob.iglob(target_dir + '/' + sf[:-1] + '/pneumonia/*'):
        pneumonia.append(filepath)

    print(f'--- {sf[:-1]} ---')
    print(f'Count of original covid19: {len(covid19)}')
    print(f'Count of original normals: {len(normals)}')
    print(f'Count of original pneumonia: {len(pneumonia)}')


    covid19_b = covid19
    normals_b = normals
    pneumonia_b = pneumonia

    for i in range( len(normals) - len(covid19) ): 
        rand_idx = np.random.randint( len(covid19) )
        covid19_b.append(covid19[rand_idx])
    for i in range( len(covid19_b) ):
        file = Path(covid19_b[i])
        shutil.copyfile(file, target_dir_balanced + '/' + sf + '/COVID-19/' + str(i) + '.png')

    for i in range( len(normals) - len(normals) ): 
        rand_idx = np.random.randint( len(normals) )
        normals_b.append(normals[rand_idx])
    for i in range( len(normals_b) ):
        file = Path(normals_b[i])
        shutil.copyfile(file, target_dir_balanced + '/' + sf + '/normal/' + str(i) + '.png')

    for i in range( len(normals) - len(pneumonia) ): 
        rand_idx = np.random.randint( len(pneumonia) )
        pneumonia_b.append(pneumonia[rand_idx])
    for i in range( len(pneumonia_b) ):
        file = Path(pneumonia_b[i])
        shutil.copyfile(file, target_dir_balanced + '/' + sf + '/pneumonia/' + str(i) + '.png')

    print(f'--- {sf} ---')
    print(f'Count of balanced covid19: {len(covid19_b)}')
    print(f'Count of balanced normals: {len(normals_b)}')
    print(f'Count of balanced pneumonia: {len(pneumonia_b)}')
