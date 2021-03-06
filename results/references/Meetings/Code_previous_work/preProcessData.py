import os
import numpy as np

import tensorflow.keras
import h5py
from keras.preprocessing.image import ImageDataGenerator

try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')

DIM_X = 95
DIM_Y = 79
DIM_Z = 60
RGB = 3
BATCH_SIZE=32
SKIP_LAYERS = 10
LIMIT_LAYERS = 70
    
FILE_NAMES = []
FILENAME_LABELS = []

ALLOWED_LABELS = ['AD','CN','MCI']
DIR_PATH = os.getcwd() + '\\data\\'


def load_all_nii_data(path, target):
    # Load dir
    target_path = path + target + "\\"
    files = os.listdir(target_path)
    
    # Allocate data matrix  (156,60, 95, 79, 3)
    data=np.zeros((len(files),DIM_Z,DIM_X, DIM_Y, RGB))
    
    # Add labels and filenames to list
    FILENAME_LABELS.extend(extract_label(files))
    FILE_NAMES.extend(files) 

    # Iterate filenames and add to data (Not tested)
    for f_i in range(0,len(files)):  
        pet_img = nib.load(target_path+files[f_i]).get_fdata() # Load image
        pet_img = pet_img[:,:,SKIP_LAYERS:LIMIT_LAYERS].T # Transpose into correct dimensions
        data[f_i] = np.broadcast_to(pet_img[...,None],pet_img.shape+(RGB,)) # Adding extra dimension to data
        
    print(f'{target} - Data shape: {str(data.shape)}')
    return data

def extract_label(files):
    return [ALLOWED_LABELS.index(string.split("_",1)[0]) for string in files]


def feature_wise_normalize(x, y):
    print("Start feature_wise_normalisation...")
    x_transformed = np.zeros((x.shape[0], DIM_Z, DIM_X, DIM_Y, RGB)) # Allocate matrix (468, 60, 95, 79, 3)

    # Reduce one dimension an extract one image
    for t_s, train_x in enumerate(range(0, x.shape[0]):
        # Train on each layer therefor dim_z is now the outer index
        x_transformed_one = np.zeros((DIM_Z, DIM_X, DIM_Y, RGB)) # Allocate transformed matrix (60, 95, 79, 3)
        

        # Write one image to training
        train_x = x[t_s]
        train_y = np.array([y[0]]*len(train_x))
        
        # Featyrewise normalisation?
        datagen_featurewise_mean = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        datagen_featurewise_mean.fit(train_x)
        
        count = 0
        # Create batches?
        for batches,(x_batch,y_batch) in enumerate(datagen_featurewise_mean.flow(train_x, train_y, shuffle=False),start=1):
            
            # No clue what this line is supposed to do..
            for i_inb in range(0, x_batch.shape[0]):
                x_transformed_one[count + i_inb, :, :, :] = (x_batch[i_inb] + 3) / 12 # Not sure what 3 and 12 is..

            count += x_batch.shape[0]
            if batches >= len(train_x) // 32: break # Since not float why not remove decimals..


        # collect x_transformed_one into x_transformed
        x_transformed[t_s] = x_transformed_one
    print(f"Normalised shape: {x_transformed.shape}")
    return x_transformed

def one_hot():
    return tensorflow.keras.utils.to_categorical(np.array(FILENAME_LABELS), len(ALLOWED_LABELS))
    
ad_data = load_all_nii_data(DIR_PATH, 'AD')
cn_data = load_all_nii_data(DIR_PATH, 'CN')
mci_data = load_all_nii_data(DIR_PATH, 'MCI')

data = np.concatenate((ad_data, cn_data, mci_data), axis=0)
data_labels = one_hot()
data_normalized = feature_wise_normalize(data, data_labels)

print(data_normalized.shape)
print(data_labels.shape)

print("Create h5 file...")
with h5py.File('data_cat4_all.h5', 'w') as hf:
    hf.create_dataset('norm_data', data=data_normalized)
    hf.create_dataset('lbl_data', data=data_labels) 

print("Print filenames to file...")
with open('file_names_cat4_all.csv', 'w') as f:
    for item in FILE_NAMES:
        f.write("%s\n" % item)
