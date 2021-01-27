import pandas as pd
import yaml
import os
import shutil

def list_to_pandas(input_list, columns=None):
    return pd.DataFrame(input_list,columns=columns)

def split_dataset(files:list, split_ratio={'train':0.70,'validation':0.15, 'test':0.15})-> (list,list,list):
    DATASET_SIZE = len(files) 
    set_sizes = {
        'train':int(DATASET_SIZE*split_ratio['train']),
        'validation':int(DATASET_SIZE*split_ratio['validation']),
        'test':int(DATASET_SIZE*split_ratio['test'])
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    
    return train_set, validation_set, test_set


def convert_df_types(
    input_df,
    types={
        'float':[],
        'int':[],
        'str':[],
        'cat':[],
        'datetime':[],
    },
    show_output=True):
    
    converter = {
        'float':'float',#pd.to_numeric,
        'int':'int',
        'str':'string',#pd.to_string,
        'cat':'category',#pd.Categorical,
        'datetime':'datetime64[ns]',#pd.to_datetime,
    }
    if len(types.values()) and show_output:
        print("Processing type of:")
    for key,cols in types.items(): # get types
        t = converter.get(key)
        for col in cols:
            if show_output: print(f"\t {t}: {col}")
            input_df[col] =  input_df[col].astype(t)
            
    return input_df

def copy_file(src, dest)-> bool:
    "Copy file from source dir to dest dir. Note that the path must exist where folders should be placed!"
    assert os.path.exists(src), "Source file does not exists"
    
    if not os.path.exists(dest):
        shutil.copy(src, dest)
        return True
    return False