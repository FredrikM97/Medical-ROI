import os

import pandas as pd
from . import utils, load
import numpy as np
from .plots.display import display_dict_to_yaml
import enum
from dataclasses import dataclass
import nibabel as nib

__all__ = ["Adni"]

@dataclass
class AdniPaths:
    root:str
    meta:str=None
    raw:str=None
    processed:str=None
    category:str=None
        
    def __post_init__(self): 
        self.meta=self.root + self.meta
        self.raw=self.root + self.raw
        self.processed=self.root + self.processed
        self.category=self.root+self.category

    def get(self,name=None):
        if name:
            return self.__dict__[name]
        return self.__dict__
    
class AdniProperties:
    filename_raw:list = None
    filename_processed = None
    filename_columns = None
    projectIdentifier:str = 'ADNI'
    files:list = []
    meta:list = []
    path:AdniPaths
    
class Adni(AdniProperties):
    
    def __init__(self, rootdir=None, metadir=None,
                 rawdir=None,processeddir=None, 
                 filename_raw=None,filename_processed=None,
                 filename_category=None,images_category=None,use_processed=False):
        self.use_processed=use_processed
        self.filename_raw=filename_raw 
        self.filename_processed=filename_processed
        self.filename_category=filename_category
        self.path = AdniPaths(
            root=rootdir, 
            meta=metadir,
            raw=rawdir,
            processed=processeddir,
            category=images_category
        )
        
    def load_meta(self, path=None, show_output=True) -> iter:
        "Load meta to list. Creates a iterator"
        path = path if path else self.path.meta
        files = [utils.xml2dict(root.findall('./*')[0], delimiter='') for root in load.load_xml(path)]
        if show_output: print(f'Root path: {path}\nLoaded files: {len(files)}')
        return files
        
    def load_files(self,path=None, columns=None, show_output=True, use_processed=None) -> iter:
        "Load image paths from image_dir"
        use_processed = use_processed if use_processed else self.use_processed 
        
        if path and columns: # Categorised
            (path,columns,func) = (
                path,columns,
                utils.split_custom_filename
        ) 
        elif use_processed: # Preprocessed
            (path,columns,func) = (
                self.path.processed,
                self.filename_raw,
                self.info_from_raw_filename 
        ) 
        elif not use_processed: # Raw
            (path,columns,func) = (
                self.path.raw,
                self.filename_raw,
                self.info_from_raw_filename 
            )
        
        print((path,columns,func))
        files =  [
            dict(
                zip(columns,[*func(filename, sep='_'), filename, path+filename])
            ) 
            for path, dirs, files in os.walk(path) 
            for filename in files if filename.endswith('.nii')
        ]
        if show_output: print(
            f"Root path: {path}\
            \nLoaded files: {len(files)}\
            \nColumns:\n\t" \
            + '\n\t'.join(columns)
        )
        return files

    def load(self,show_output=True) -> None:
        "Load both images and meta"
        self.files = self.load_files(show_output=show_output)
        self.meta = self.load_meta(show_output=show_output)
    
    def info_from_raw_filename(self,filename, sep=None) -> str:
        "Get all info from filename instead (bit slower and could do wrong but removes need of multiple folders)"
        assert '_br_raw_' in filename, "The imported filenames does not contain the expected split: '_br_raw_'"
        if len(filename.split('_br_raw_')[1].split('_')) == 3:
            return self.info_from_raw_filename_no_image_number(filename)
        
        split_order = [('_',1),('_',3),('_',1),('_br_raw_',1),('_',1),('_',1),('_',1),('.',1)] 
        c =[]
        def split(strng, sep, pos):
            strng = strng.split(sep)
            return sep.join(strng[:pos]), sep.join(strng[pos:])
        if self.use_processed: filename = utils.remove_preprocessed_filename_definition(filename)
        i = filename
        for s in split_order:
            e,i = split(i, s[0],s[1])
            c.append(e)
        return c
    
    def info_from_raw_filename_no_image_number(self, filename) -> str:
        "Get all info from filenames where no image number is included"
        split_order = [('_',1),('_',3),('_',1),('_br_raw_',1),('_',1),('_',1),('.',1)] 
        c =[]
        def split(strng, sep, pos):
            strng = strng.split(sep)
            return sep.join(strng[:pos]), sep.join(strng[pos:])
        if self.use_processed: filename = utils.remove_preprocessed_filename_definition(filename)
        i = filename
        for s in split_order:
            e,i = split(i, s[0],s[1])
            c.append(e)
        c.insert(5, '')
        return c
        
    def get_files(self, path=None,columns=None) -> iter:
        "Get files from class"
        files = self.load_files(path=path,columns=columns) if path and columns else self.files   
                
        return files
            
    def load_images(self, files=None) -> iter:
        "Load image into memory"
        files = (file['path'] for file in (files if files else self.files))
        print(list(files))
        return (nib.load(file).get_fdata() for file in files)
    
    def get_dataset(self, dist:list=[0.6, 0.15])-> (list,list, list):
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
    
    def to_slices(self, image_list=None) -> iter:
        "Get each slice from each image in path"
        image_list = image_list if image_list else self.load_images()
        
        def func(img_array):
            for image in img_array:
                for slices in image:
                    for one_slice in slices:
                        yield one_slice
                
        return self.to_array(func(image_list=image_list))
            
    def to_array(self, image_list=None) -> iter:
        "Convert images into numpy arrays and transpose from (x,y,z,n) -> (n,z,x,y)"
        image_list = image_list if image_list else self.load_images()
        func = lambda file: file.get_fdata().T

        return (func(file) for file in image_list)
    
    def to_df(self, show_output=True):
        "Convert image list and meta list to "
        files_df = self.files_to_df(show_output=show_output)
        meta_df = self.meta_to_df(show_output=show_output)
        
        df = pd.merge(files_df,meta_df, on=['subject.subjectIdentifier','subject.study.imagingProtocol.imageUID'])
        return df
    
    def meta_to_df(self, show_output=True):
        "Convert metadata list to dataframe"
        meta_df = pd.DataFrame(self.meta).sort_values('subject.subjectIdentifier')
        # Add I so that images and meta is named the same!
        meta_df['subject.study.imagingProtocol.imageUID'] = 'I'+meta_df['subject.study.imagingProtocol.imageUID'].astype(str)
        
        meta_df = utils.df_object2type(meta_df, types={
            'cat':['subject.researchGroup'],'datetime':['subject.study.series.dateAcquired' ]}, show_output=show_output)

        return meta_df
    
    def files_to_df(self, show_output=True):
        "Convert files list to dataframe"
        files_df = pd.DataFrame(self.get_files(),columns=self.filename_raw)
        files_df = utils.df_object2type(files_df, types={},show_output=show_output)
        
        return files_df