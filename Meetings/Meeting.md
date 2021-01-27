# Meeting 1 (2020-11-09)
## TODO
Intensity normalization has limitations when applied to medical images
    Try: Define different regions w.r.t intensity, normalize earch region separately or attentive normalization

Instead of hole brain, identify ROIs across different 3D layers of brain scan.

## Meetings
10-10:30 or 10:30-11:00

## Task
Idea: Problem definition, literature review, project plan -> 5 pages
Idea: Read and add to report
Idea: Define phrasing of contribution "what are we gonna do?"
Structure: Read 5-10 pages of literature and add to report
Deadline: 10'th december (supervisors 25'th november)
Goal: Going to do, implementation, evaluation (why is this going to work and compared to what?)

# Meeting 2 (2020-11-16)
* We are only using PET scans not DAT scans
* Use images from Quantifying brain metabolism from FDG-PET to do visualisation can be good
* PET scans typically focus on glucose (sugar) metabolism and DaT/SPECT scans focus on the activity of the dopamine transporter.
* Stick to pet scans due to guidelines?

## Project description
Find which parts are interested
Images not 3D in reality but stacked on each other
Find which slices "levels" is most interested
Then highlight and extract these areas in the brain slices
Which parts is affected by the disorder and which slices is more important than the others.
Achievement: Higher accuracy with lower performance needed compared to other reports that utilises hole brain. Not just regions but which slices is good to use.

# Meeting 3 (2020-11-23)
Relate to what have been done
## Aim 
* Questions should be more detailed not abstract
* Where is missing knowledge challenge?
* What is challenging in algorithmic point of view
* Are there relations between regions?
* List which evaluation and visualisation methods that is being used.
## Problem definition
* Evaluate regions related to each other?
* Testing different type of loss functions
* Apply more ROI as input at once possible?
* If valve filter then reason why is it added?

## Limitations
* Dataset
* Lots of limitations in evaluation
* Images or clinical evaluation? Clinical evaluation?
* Only using PET scans
* Only european brains? No asian brains
* If you dont know it then write it down
* Limited to ADNI and their dataset
* TL;DR: Something that exists but not covered in report

## Questions
* Do we need to include accknowledge of adni dataset?
	- No, not until it is published but good to refer to them.
* Are we using PET or PET/CT which stand for computed tomography?
	- We are using 18F-FDG-PET only!
* Do we want to exclude layers or slices before sending them as input to NN?
	- Free to choose if we want to do ROI within or outside.
* What week is half-time seminar (deadlines?)
	- End of feburary. Send mail to Slawomir.
* To what extent should AD be explained pathologically?
	- Not needed to be done here. Not the focus of thesis.
	
# Meeting 4 (2020-11-30)
# Introduction
What have been done?
# Aim 
* Quite similar and could be combined.
* A bit specific
* No yes/no questions
# Limitations
* More meat on the limitation. Why is this a limitation? 3-4 sentences

# Literature review
* Dont limit to one PET scans or other disorders
* Go from other PET scans to PET scans to cover greater
* Write about each report in a structure to understand them give compare/relate the advantages/disadvantages

# Project plan
* Very detailed
* Categorise steps in bigger steps (Literature (learning), implementation, Evaluation, report writing)
* Divide project into five steps 
* Little description but not to much

* Use dementia instead of disease

# Meeting 5 (2020-12-07)

# Metric
* Structure of data and how the dataset looks like

## Aim
* Question number one should be reformulated.
* Relate it to RPN. Augmenting RPN to be applied to an unsupervised method.
* Question number two. What impact compared to previous presented methods
* Question three hard to understand for someone who dont know the domain. Why is it interesting?
How does the region affect the classification result? Is any of the regions domenant? How is their relation? (Not studied before)

# Limitations
* Some highlights: European, American, Asian brain images could be biased.

# Meeting 6 (2021-01-11)
* Present visualisation until next week
* Sent mail to slawomir of deadlines
* Investigate if matlab code can be converted to python (Max two weeks)
* Contact radiologist of what is important: Precision (Area) vs Accuracy? Source or area around the area?
* Metrics for evaluation?
* Investigate if predefined models to apply for CNN
* Is pytorch better than keras?
* How to build framework for visualisation?
* Is tensorboard compatible to pytorch or alternatives?
* Note: Use generators for images if we need to iterate. Proven to be faster

# Meeting 7 (2021-01-25)
* Feature-wise normalization is due to existance of negative values
* Each scan session produces 6 .nii images
	- Options discussed: Pick one, keep all, average. ADNI has average as an option for download.
	- No real answer, suggest pick 1 of them. Early scans only feature 1 image, so using could cause imbalance.
		- Need to explore in detail the differences between each image from the same session

* How to input to CNN?
	- We suggested mosaic, one slice at a time, layers as channels
	- Not required, 3D CNN should handle. If not re-evaluate above options
	- Depth of kernel important. 3x3x3 vs 3x3x1

* MMSE Score (and other scores) used when validating models. Part of the clinical assessment of subjects, before a potential diagnosis.

* Only use data of ADNI3 - at least at first. Each phase uses slightly different protocols, resulting in some potential incompatability.

* .nii metadata
	- Does .nii files contain a header with meta data?
	- How does SPM treat .nii file?
		- File name is retained, but prepended with 'iw'
		- Metadata?
