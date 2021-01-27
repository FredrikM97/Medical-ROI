import os
import xml.etree.ElementTree as ET
import nibabel as nib
import pandas as pd
from . import misc_util
import numpy as np
from .display import display_dict_to_yaml
import enum
    
class ADNI_PATHS:
    processed=None
    raw=None
    category=None
    meta=None

    def update_paths(self,rootpath):
        self.processed=rootpath+'/processed/'
        self.raw=rootpath+'/adni_raw/'
        self.category={
            'root':rootpath+'/adni/',
            'AD':rootpath+'/adni/' + 'AD/',
            'CN':rootpath+'/adni/' + 'CN/',
            'MCI':rootpath+'/adni/' + 'MCI/',
        }
        self.meta=rootpath+'/meta/'
        
class ADNI_PROPERTIES:
    columns:list = [
            'projectIdentifier', 
            'subject.subjectIdentifier',
            'subject.study.series.modality',
            'subject.study.imagingProtocol.description',
            'subject.study.series.dateAcquiredPrecise', 
            'image_nbr',
            'series',
            'subject.study.imagingProtocol.imageUID',
            'filename',
            'path',
        ]
    projectIdentifier:str = 'ADNI'
    files:list = []
    meta:list = []
    category_cols =[
        'subject.researchGroup',
        'subject.subjectIdentifier',
        'subject.study.imagingProtocol.imageUID',
        'image_nbr',
        'filename',
        'path',
    ]
    category_filename = [
        'subject.researchGroup',
        'subject.subjectIdentifier',
        'subject.study.imagingProtocol.imageUID',
        'image_nbr',
    ]
    path = ADNI_PATHS()
    
class ADNI(ADNI_PROPERTIES):
    
    def __init__(self, root_dir):
        self.path.update_paths(root_dir)
        
    def load_meta(self, path=None, show_output=True) -> iter:
        "Load meta to list. Creates a iterator"
        path = path if path else self.path.meta
        files = [self.__xml_to_dict(root.findall('./*')[0], delimiter='') for root in self.__load_xml(path)]
        if show_output: print(f'Root path: {path}\nLoaded files: {len(files)}')
        return files
        
    def load_files(self,path=None, columns=None, show_output=True) -> iter:
        "Load image paths from image_dir"
        (path,columns,func) = (
            path,columns,
            self.info_from_custom_filename
        ) if path and columns else (
            self.path.raw,
            self.columns,
            self.info_from_raw_filename
        )
        
        start = path.rfind(os.sep) + 1
        
        getFolder = lambda path: path[start:].split(os.sep)
        
        files =  [
            dict(
                zip(columns,[*func(file), file, path+"/"+file])
            ) 
            for path, dirs, files in os.walk(path) 
            for file in files if file.endswith('.nii')
        ]
        if show_output: print(f"Root path: {path}\nLoaded files: {len(files)}\nColumns:\n\t" + '\n\t'.join(columns))
        return files

    def load(self,show_output=True):
        "Load both images and meta"
        self.files = self.load_files(show_output=show_output)
        self.meta = self.load_meta(show_output=show_output)
    
    def info_from_custom_filename(self, file, sep='#'):
        slices= file.split(sep)
        slices[-1] = slices[-1].split(".")[0]

        return slices
    
    def info_from_raw_filename(self,file):
        "Get all info from filename instead (bit slower and could do wrong but removes need of multiple folders)"
        split_order = [('_',1),('_',3),('_',1),('_br_raw_',1),('_',1),('_',1),('_',1),('.',1)] 
        c =[]
        def split(strng, sep, pos):
            strng = strng.split(sep)
            return sep.join(strng[:pos]), sep.join(strng[pos:])
        i = file
        for s in split_order:
            e,i = split(i, s[0],s[1])
            c.append(e)
        return c
        
    def get_files(self, path=None,columns=None) -> iter:
        "Get files from class"
        files = self.load_files(path=path,columns=columns) if path and columns else self.files   
        
        def _generator():
            for file in files:
                yield file
                
        return _generator()
            
    def get_images(self, files=None) -> iter:
        "Load image file into memory"
        files = files if files else self.files

        def _generator():
            for file in files:
                yield self.get_image(file['path'])
                
        return _generator()
        
    
    def get_metas(self) -> iter:
        "Get metdata"
        for meta in self.meta:
            yield meta
    
    def get_dataset(self, dist:list=[0.6, 0.15]):
        "Load dataset and assign labels. Double check that labels always are the same!"
        
        df = self.to_df()
        df["labels"] = df["subject.researchGroup"].cat.codes
        dataset = df[['path', 'labels']]

        DATASET_SIZE = len(df)
        train, validate, test = np.split(dataset.sample(frac=1), [int(dist[0]*DATASET_SIZE), int((1-dist[1])*DATASET_SIZE)])
        print(
            f"Dataset sizes:\n\tTrain: {len(train)}({dist[0]})\
            \n\tValidation: {len(validate)}({dist[1]})\
            \n\tTest: {len(test)}({dist[1]})\
            \n\tTotal: {DATASET_SIZE}"
        )
        return train, validate, test
        
    def get_dataset_images(self):
        pass
    
    def to_slices(self, image_list=None) -> iter:
        "Get each slice from each image in path"
        image_list = image_list if image_list else self.get_images()
        
        def _generator():
            for image in self.to_array(image_list=image_list):
                for slices in image:
                    for one_slice in slices:
                        print("derp",one_slice.shape,slices.shape)
                        yield one_slice
                
        return _generator()
            
    def to_array(self, image_list=None) -> iter:
        "Convert images into numpy arrays and transpose from (x,y,z,n) -> (n,z,x,y)"
        image_list = image_list if image_list else self.get_images()

        def _generator():   
            for file in image_list:
                yield file.get_fdata().T
                
        return _generator()
    
    def to_df(self, show_output=True):
        "Convert image list and meta list to "
        files_df = self.files_to_df(show_output=show_output)
        meta_df = self.meta_to_df(show_output=show_output)        
        
        df = self.__merge_df(files_df,meta_df)
        return df
    
    def meta_to_df(self, show_output=True):
        "Convert metadata list to dataframe"
        meta_df = pd.DataFrame(list(self.get_metas())).sort_values('subject.subjectIdentifier')
        # Add I so that images and meta is named the same!
        meta_df['subject.study.imagingProtocol.imageUID'] = 'I' + meta_df['subject.study.imagingProtocol.imageUID']
        
        meta_df = misc_util.convert_df_types(meta_df, types={
            'float':[
                'subject.study.subjectAge',
                'subject.study.weightKg',
                'subject.visit.assessment.component.assessmentScore_MMSCORE',
                'subject.visit.assessment.component.assessmentScore_GDTOTAL',
                'subject.visit.assessment.component.assessmentScore_CDGLOBAL',
                'subject.visit.assessment.component.assessmentScore_NPISCORE',
                'subject.visit.assessment.component.assessmentScore_FAQTOTAL'
            ],
            'cat':[
                'subject.researchGroup'
            ],
            'str':[
                'subject.subjectSex'
            ],
            'datetime':[
                'subject.study.series.dateAcquired'
            ]
        }, show_output=show_output)
        return meta_df
    
    def files_to_df(self, show_output=True):
        "Convert files list to dataframe"
        files_df = pd.DataFrame(list(self.get_files()),columns=self.columns)
        files_df = misc_util.convert_df_types(files_df, types={},show_output=show_output)
        
        return files_df
    
    def get_image(self, path):
        "Load one image"
        return nib.load(path)
    
    def get_meta(self, path):
        "Get metadata from path"
        return self.load_meta(path)
    
    def save_to_category(self,output_df, path:dict=None, show_output=True)->None:
        "Save images to categories based on parameters from dataframe"
        path = path if path else self.path.category
        
        if show_output: print(f"Copy files: \n\t"+ "\n\t".join([f'{key}: {value}' for key, value in path.items()]))
        stats = {
            'AD':{
                'Skip':0, # Skip
                'Transfer':0, # Transfer
            },
            'CN':{
                'Skip':0, # Skip
                'Transfer':0, # Transfer
            },
            'MCI':{
                'Skip':0, # Skip
                'Transfer':0, # Transfer
            }
        }
        conv = {
            0:'Skip',
            1:'Transfer'
        }
        def inner(row):
            filename = f"{'#'.join([row[p] for p in self.category_filename])}.nii"
            response = misc_util.copy_file(str(row['path']), f"{path[row['subject.researchGroup']]}/{filename}")
            stats[row['subject.researchGroup']][conv[response]] += 1 
        
        output_df.apply(lambda row: inner(row), axis=1)
        
        if show_output: display_dict_to_yaml({'Statistics':stats})
   
        
    def __xml_to_dict(self,r, parent='', delimiter=".") -> list:
        "Iterate through all xml files and add them to a dictionary"
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
                return {}
        return recursive(r, parent=parent, delimiter=delimiter)
    
    def __load_xml(self, path:str):
        "Load XML from dictory and return a generator"
        assert path, "No path defined"
        for filename in os.listdir(path):
            if not filename.endswith('.xml'): continue
            fullname = os.path.join(path, filename)
            yield ET.parse(fullname)
            
    def __merge_df(self,image_df,meta_df, cols=['subject.subjectIdentifier','subject.study.imagingProtocol.imageUID']):
        """Merge two dataframes based on 'subject.subjectIdentifier' and 'subject.study.imagingProtocol.imageUID'"""
        return image_df.merge(meta_df,on=cols)