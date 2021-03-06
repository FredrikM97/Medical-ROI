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
2. Rewrote code related to ADNI images. 
3. Testing of SPM 
4. Tried to find alternative to SPM.
    * Found wrapper for SPM in python. Might be easier to use SPM to save time.

## TODO:
1. Add infinity generator (Assume Keras/Pytorch need it)
2. Implement CNN network 
3. Add so each dataset load images


# Meeting 2 (2021-01-29)

## Questions
1. What should be logged? 
	* Loss, metrics for train, validation and test? 
	* Images, and what kind of images? 

2. Anything we should keep an eye on? 
3. How should we split data? 
	* Train, val, test (0.7,0.15,0.15)? 
	* Testset: Manually extract images or random? 

4. 259 Images (unique subjects, latest nii files) 
	* More data needed for DL?
	* Augmentation,bad to fix it?
	* Pretrained model? 

## Finished
1. Implemented framework in pytorch
2. Implemented dataloader (waiting for preprocess)
3. Implemented CNN network (waiting for dataloader) 
4. Fixed categorise and analyse SPM images. 

## Todo
1. Add split for datasets (Waiting for dataloader) 
2. Fix CNN 
3. Add tensorboard for logging
4. Visualise activation map? 

# Meeting 3
## Questions
1. Should we change size of input or is 79,95,79 good?

## Finished
1. Support for tensorboard
2. Trained on test model.

## Todo
1. Implement VGG16?
2. What to do after activation map? Scientific reports? Combine something?
