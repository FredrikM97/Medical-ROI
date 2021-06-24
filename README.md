# Prioritization of Informative Regions in PET Scans for Classification of Alzheimer’s Disease
Master thesis

## Workflow
Examples of the approach for each aspect of the project are presented in the Jupyter notebooks. Each file consists of XX_Y, where XX is the index of development and Y is a subcategory of the index where the methodology is split into smaller pieces. A summary of the main focus is divided into sections; Data, WSOL, BBOX, and AAL. 

## Configuration
To configure the models, take a look into the conf folder where a base config in config/base decides what the default parameters to use. To overwrite them define another config with the same structure, which will overwrite the existing. Remember to keep the indices correct. 

Example:

{
"name":"debug",
"model":{
    "arch": {
        "name": "resnet18_brew",
        "args": {}
    }
},

The name will be overwritten for the classifier and arch name too.

## Data
In order to run the project, image data is required with the format .nii. The structure requires to be included in the data folder (but can be overwritten in the config), and the files need the structure class/filename.nii where the class is the target class of an image.

