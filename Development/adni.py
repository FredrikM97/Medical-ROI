import os
import xml.etree.ElementTree as ET
import nibabel as nib

class ADNI:
    columns = [
        'projectIdentifier',
        'subject.subjectIdentifier',
        'ImageProtocol.description',
        'dateAcquired',
        'subject.study.imagingProtocol.imageUID', 
        'filename',
        'path'
    ]
    projectIdentifier = 'ADNI'
    files = []
    meta = []
    
    
    def __init__(self, root_dir):
        self.preprocessed_image_dir = root_dir + '/processed/'
        self.raw_image_dir = root_dir + '/adni_raw/'
        self.image_dir = root_dir + '/adni/'
        self.meta_dir = root_dir + '/meta/'
    
    def load_meta(self, path=None) -> iter:
        "Load meta to list. Creates a iterator"
        if not path: path=self.meta_dir
        self.meta = [self.__xml_to_dict(root.findall('./*')[0], delimiter='') for root in self.__load_xml(path)]
        
    def load_files(self,path=None) -> iter:
        "Load image paths from image_dir"
        if not path: path=self.raw_image_dir
        contents = []
        start = path.rfind(os.sep) + 1
        
        getFolder = lambda path: path[start:].split(os.sep)
        
        self.files =[
            dict(
                zip(
                    self.columns,[
                        self.projectIdentifier,*getFolder(path), file, path+"/"+file
                    ]
                )
            ) 
            for path, dirs, files in os.walk(path) 
            for file in files if file.endswith('.nii')
        ]
    def load(self):
        "Load both images and meta"
        self.load_files()
        self.load_meta()
    def get_files(self, files=None) -> iter:
        for file in files:
            yield file
            
    def get_images(self, files=None) -> iter:
        if not files: files = self.files
        #ADNI_002_S_0295_PT_ADNI_Brain_PET__Raw_FDG_br_raw_20110609102421118_60_S111104_I239487.nii
        for file in files:
            yield nib.load(file['path'])
    
    def get_metas(self) -> iter:
        for meta in self.meta:
            yield meta
    
    def to_slices(self, image_list=None) -> iter:
        if not image_list: image_list = self.get_images()
        for image in self.to_array():
            for slices in image:
                for one_slice in slices:
                    print("derp",one_slice.shape,slices.shape)
                    yield one_slice
            
    def to_array(self, image_list=None) -> iter:
        if not image_list: image_list = self.get_images()
        for file in image_list:
            yield file.get_fdata().T
            
    def to_df(self):
        "Convert image list and meta list to "
        image_df = pd.DataFrame(list(self.get_files()),self.columns)
        
        meta_df = pd.DataFrame(list(self.meta)).sort_values('subject.subjectIdentifier')
        # Add I so that images and meta is named the same!
        meta_df['subject.study.imagingProtocol.imageUID'] = 'I' + meta_df['subject.study.imagingProtocol.imageUID']
        
        df = self.__merge_df(image_df,meta_df)
        return df
    
    def get_image(self, path):
        return nib.load(path)
    
    def get_meta(self, path):
        return 
    def nii_header(self,image_obj):
        return image_obj.header
    
    def save_to_category(self,output_df, path=None)->None:
        "Save images to categories based on parameters from dataframe"
        if not path: path=self.image_dir
        output_df.apply(lambda row: copy_file(str(row['path']), f"{path}/{row['subject.researchGroup']}/{row['filename']}"), axis=1)
        
    
        
    def __xml_to_dict(self,r, parent='', delimiter=".") -> list:
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
    
    def __load_xml(self, path:str):
        "Load XML from dictory and return a generator"
        assert path, "No path defined"
        content = []
        for filename in os.listdir(path):
            if not filename.endswith('.xml'): continue
            fullname = os.path.join(path, filename)
            yield self.__get_xml(fullname)
            
    def __get_xml(self, path):
        return ET.parse(path)
            
            
    def __merge_df(self,image_df,meta_df, cols=['subject.subjectIdentifier','subject.study.imagingProtocol.imageUID']):
        """Merge two dataframes based on 'subject.subjectIdentifier' and 'subject.study.imagingProtocol.imageUID'"""
        return image_df.merge(meta_df,on=cols)