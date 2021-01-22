# Meeting 1 (2021-01-20)
* Improve data loading, some bugs discovered
* Evaluate if separate python script instead of SPM
* Limitations of matlab, one file at a time, can it handle multiple files?
    - with supplied script, yes

## Questions:
1. Some folders contain 6 nii files. How to treat:
    - Keep all
    - Select 1
    - Average
    * Example:
        * ADNI_002_S_1261_PT_ADNI_Brain_PET__Raw_FDG_br_raw_20110301121631780_456_S100595_I221695
        * ADNI_002_S_1261_PT_ADNI_Brain_PET__Raw_FDG_br_raw_20110301121638249_191_S100595_I221695
        * ADNI_002_S_1261_PT_ADNI_Brain_PET__Raw_FDG_br_raw_20110301121642624_127_S100595_I221695
    
2. Input suggestions (CNN):
    * Mosaik one image as input.
    * Each slice as an input
    * One slice as one channel (replacement instead of color).
    
3. Should we share the github repository? If yes, we need an mail in order to invite.

4. Metric and visualisation:
    * Are the metrics (ex: MMSCORE) to know severity of disorder in brain? How can we use it?
    * Do we gain anything to do more plots of the brain (slices etc..) or should we more to CNN?

5. Related to code from previous work (beginning of project):
    * Can we use SPM as preprocessing directly? Code that we got did feature_wise_normalize. Why?
    * Is it a good idea to use the provided CNN and go directly to metric implementation and ROI?
    
## What we have done:
1. We fixed plots for the different metrics. 
2. Rewrote all code for ADNI. Easier to follow notebook code/reduced redundancy.
    * Added function import of images and categorise them into AD,MCI, CN
    * Added function to load of image to generators (performance)
    * Added function to get one slice at a time from generator
    * Added function to split images and labels into train, validation and test dataset
3. Testing of SPM 
4. Tried to find alternative to SPM.
    * Found wrapper for SPM in python. Might be easier to use SPM to save time.

## TODO:
1. Add infinity generator (Assume Keras/Pytorch need it)
2. Implement CNN network 
3. Add so each dataset load images
