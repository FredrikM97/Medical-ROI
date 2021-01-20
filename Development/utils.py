import pandas as pd
import yaml

def downloadContent():
    pass

def getLabels():
    pass


def list_to_pandas(input_list, columns=None):
    return pd.DataFrame(input_list,columns=columns)



def convert_df_types(
    input_df,
    types={
        'num':[],
        'str':[],
        'datetime':[],
    },
    datetime_format='%Y-%m-%d',
    ):
    
    converter = {
        'num':{
            'obj':pd.to_numeric,
            'params':{}
        },
        'str':{
            'obj':pd.DataFrame.to_string,
            'params':{}
        },
        'datetime':{
            'obj':pd.to_datetime,
            'params':{'format':datetime_format}
        }
    }
    for key,cols in types.items(): # get types
        t = converter.get(key)
        for col in cols:
            print("Processing type of", col)
            input_df[col] =  t['obj'](input_df[col],**t['params'])
            
    return input_df

def merge_df(image_df,meta_df, cols=[]):
    """Merge two dataframes based on 'subject.subjectIdentifier' and 'subject.study.imagingProtocol.imageUID'"""
    return image_df.merge(meta_df,on=cols)

