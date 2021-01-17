import pandas as pd

def downloadContent():
    pass

def getLabels():
    pass
def xml2dict(r, parent='', delimiter=".") -> list:
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

def xml_dict_2_pd(xml_dict):
    """Convert a xml dictionary to pandas dataframe and process to fit metadata"""
    d = [xml2dict(root.findall('./*')[0], delimiter='') for root in xml_dict]
    
    meta_df = pd.DataFrame(d).sort_values('subject.subjectIdentifier')
    # Add I so that images and meta is named the same!
    meta_df['subject.study.imagingProtocol.imageUID'] = 'I' + meta_df['subject.study.imagingProtocol.imageUID']
    return meta_df

def convert_df_types(
    input_df,
    types={
        'num':[],
        'str':[],
        'datetime':[],
    datetime_format='%Y-%m-%d',
    }):
    
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
            'i
        }
    }
    for key,cols in types.items():
        t = converter.get(key)
        for col in cols:
            t['obj'](col,*t['params'])
            
    
    meta_df['subject.study.subjectAge'] =  pd.to_numeric(meta_df['subject.study.subjectAge']) 
meta_df['subject.study.weightKg'] =  pd.to_numeric(meta_df['subject.study.weightKg']) 
meta_df['subject.study.series.dateAcquired'] = pd.to_datetime(meta_df['subject.study.series.dateAcquired'], format='%Y-%m-%d')
    return 

def display_all_pd_cols(input_df):
    with pd.option_context('display.max_columns', None):
        display(input_df.head())