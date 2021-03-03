import os

import pandas as pd
from . import misc_util
import numpy as np
from .display import display_dict_to_yaml
import enum
from dataclasses import dataclass

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
        files = [misc_util.xml_to_dict(root.findall('./*')[0], delimiter='') for root in misc_util.load_xml(path)]
        if show_output: misc_util.default_print(f'Root path: {path}\nLoaded files: {len(files)}')
        return files
        
    def load_files(self,path=None, columns=None, show_output=True, use_processed=None) -> iter:
        "Load image paths from image_dir"
        use_processed = use_processed if use_processed else self.use_processed 
        
        if path and columns: # Categorised
            (path,columns,func) = (
                path,columns,
                misc_util.split_custom_filename
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
                zip(columns,[*func(filename), filename, path+filename])
            ) 
            for path, dirs, files in os.walk(path) 
            for filename in files if filename.endswith('.nii')
        ]
        if show_output: misc_util.default_print(
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
    
    def info_from_raw_filename(self,filename) -> str:
        "Get all info from filename instead (bit slower and could do wrong but removes need of multiple folders)"
        assert '_br_raw_' in filename, "The imported filenames does not contain the expected split: '_br_raw_'"
        if len(filename.split('_br_raw_')[1].split('_')) == 3:
            return self.info_from_raw_filename_no_image_number(filename)
        
        split_order = [('_',1),('_',3),('_',1),('_br_raw_',1),('_',1),('_',1),('_',1),('.',1)] 
        c =[]
        def split(strng, sep, pos):
            strng = strng.split(sep)
            return sep.join(strng[:pos]), sep.join(strng[pos:])
        if self.use_processed: filename = misc_util.remove_preprocessed_filename_definition(filename)
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
        if self.use_processed: filename = misc_util.remove_preprocessed_filename_definition(filename)
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
        "Load image file into memory"
        files = (file['path'] for file in (files if files else self.files))
        return misc_util.load_images(files)
    
    def get_dataset(self, dist:list=[0.6, 0.15])-> (list,list, list):
        "Load dataset and assign labels. Double check that labels always are the same!"
        
        df = self.to_df()
        df["labels"] = df["subject.researchGroup"].cat.codes
        dataset = df[['path', 'labels']]

        DATASET_SIZE = len(df)
        train, validate, test = np.split(dataset.sample(frac=1), [int(dist[0]*DATASET_SIZE), int((1-dist[1])*DATASET_SIZE)])
        misc_util.default_print(
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
                
        return misc_util.generator(self.to_array(image_list=image_list), func)
            
    def to_array(self, image_list=None) -> iter:
        "Convert images into numpy arrays and transpose from (x,y,z,n) -> (n,z,x,y)"
        image_list = image_list if image_list else self.load_images()
        func = lambda file: file.get_fdata().T

        return misc_util.generator(image_list, func)
    
    def to_df(self, show_output=True):
        "Convert image list and meta list to "
        files_df = self.files_to_df(show_output=show_output)
        meta_df = self.meta_to_df(show_output=show_output)
        
        df = misc_util.merge_df(files_df,meta_df, cols=['subject.subjectIdentifier','subject.study.imagingProtocol.imageUID'])
        return df
    
    def meta_to_df(self, show_output=True):
        "Convert metadata list to dataframe"
        meta_df = pd.DataFrame(self.meta).sort_values('subject.subjectIdentifier')
        # Add I so that images and meta is named the same!
        meta_df['subject.study.imagingProtocol.imageUID'] = 'I'+meta_df['subject.study.imagingProtocol.imageUID'].astype(str)
        
        meta_df = misc_util.convert_df_types(meta_df, types={
            'cat':['subject.researchGroup'],'datetime':['subject.study.series.dateAcquired' ]}, show_output=show_output)

        return meta_df
    
    def files_to_df(self, show_output=True):
        "Convert files list to dataframe"
        files_df = pd.DataFrame(self.get_files(),columns=self.filename_raw)
        files_df = misc_util.convert_df_types(files_df, types={},show_output=show_output)
        
        return files_df
    
    """
    def save_to_category(self,output_df, path:Disorders=None, show_output=True)->None:
        "Save images to categories based on parameters from dataframe"
        path= path if path else self.path.disorders

        func = lambda path: f"Copy files: \n\t"+ "\n\t".join([f'{key}: {path.get(key)}' for key in path.get().keys()])
        
        if show_output: misc_util.default_print(func(path))
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
            filename = f"{'#'.join([row[p] for p in self.filename_columns])}.nii"

            response = misc_util.copy_file(str(row['path']), f"{path.get(row['subject.researchGroup'])}/{filename}")
            stats[row['subject.researchGroup']][conv[response]] += 1 
        
        output_df.apply(lambda row: inner(row), axis=1)
        if show_output: display_dict_to_yaml({'Statistics':stats})
        """
    
