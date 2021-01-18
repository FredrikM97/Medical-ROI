import pandas as pd
import yaml

def downloadContent():
    pass

def getLabels():
    pass
def xml_to_dict(r, parent='', delimiter=".") -> list:
    param = lambda r:"_"+list(r.attrib.values())[0] if r.attrib else ''
    def recursive(r, parent='', delimiter=".") -> list:
        cont = {}
        # If list
        if layers := r.findall("./*"):
            [cont.update(recursive(x, parent=parent +delimiter+ x.tag)) for x in layers]
            return cont

        elif r.text and '\n' not in r.text: # get text
            return {parent + param(r):r.text}
        else:
            return {'dummy':None}
    return recursive(r, parent=parent, delimiter=delimiter)

def xml_to_pd(xml_dict:dict):
    """Convert a xml dictionary to pandas dataframe and process to fit metadata"""
    d = [xml_to_dict(root.findall('./*')[0], delimiter='') for root in xml_dict]
    
    meta_df = pd.DataFrame(d).sort_values('subject.subjectIdentifier')
    # Add I so that images and meta is named the same!
    meta_df['subject.study.imagingProtocol.imageUID'] = 'I' + meta_df['subject.study.imagingProtocol.imageUID']
    return meta_df

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

