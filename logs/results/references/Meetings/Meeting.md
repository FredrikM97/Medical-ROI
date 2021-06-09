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
## TODO:
* Finish preprocess
* Present CNN model next week

# Meeting 8 (2021-02-01)
* Only use data of ADNI1. Exclude subjects with MCI that never developed AD - list can be found by searching 'MCI conversion' - all subjects in this file eventually developed AD. Reasoning?

* Cross-validation - k: {8, 9, 10} supposedly good. Re-run lots of times, present average + std

* ~300 samples should be enough 

* Confidence interval.

* Randomize testset

* Wait with transfer-learning etc.

## TODO
* Create basic model. (3D CNN and sailency guided)
* Activation map until next week


# Meeting 9 (2021-02-09)
* Exploratory Data Analysis
	* Study in detail how networks responds to each class, to make sure the model actually identifies something interesting before moving on to ROI
	* Techniques: Saliency Map, CAM, t-SNE

* Spend at most 2-3 more weeks with developing a good CNN. If no good results, Amira should have a good model to hopefully share.
	* VGG-16 ?

* A good thesis often have a comparative part
	* Could be input: Individual slices, mosaic, full 3D
	* Compare R-CNN, Fast, Faster, Mask in medical domain

* Comparing suggested ROIs with actual anatomical regions of the brain
	* If the suggested ROI stretch over multiple anatomical regions, is it less reliable?
		* Compare overlap for regions from a correct decision to that of a incorrect decision

## TODO:
* Ask Amira about GitHub invite
* Continue CNN + EDA, techniques for exploring the model - saliency map, t-SNE

# Meeting 12 (2021-03-08)
* Permanent ROI
	- Motivated:
		* Computationally less exhaustive
		* Clinically, radiologists don't look at the entire image either [source?]

	- Different ways of extracting ROIs, perform some comparative analysis
	 	* Hard atlas (anatomical regions)
		* Find discriminative regions through ML
			- Different weakly supervised techniqes
				* CAM, saliency etc

* Adjust plots:
	- Average brain reference image behind saliency map plot
	- Create one average plot for every class, of all instances where the class was correctly predicted
	- Also include some randomly chosen mis-classified sample
	- Flip CAM plots and change to (axial?) view (top-down)

* Increase discussion about model selection.
	- And then present final model in results

* Discussion section
	- Saliency results matches [Ding]

# Meeting 13 (2021-03-22)
* Half-time presentation:
	* Meeting notes in slides

* Physical regions, extract using atlas - AAL [Xiaoxi]

* Main approach suggestions:
	* Keep BBR in final network: Learn to construct BB from input image, without having to produce CAM. During training, CAM is used as label. Results in different BB for different input images.
	* Construct global average heatmap (CAM, etc), and extract permanent ROIs. Then train a new classifier to predict using only those ROIs. Results in same BB for different input images.

* Make a simple initial complete solution
	* Segmentation
	* Expect bad results, backtrack

* Investigate BBR immediately on intensity, rather than segmented image

* This week:
	* Presentation
	* Construct 9 global average CAMs
	* Construct average CAM for same input image, over multiple attempts
	* Refine code structure

# Meeting 14 (2021-03-29)
* Slawomir feedback:
	- Model excessively complex
	- Improve motivation of project

* Try to make model simpler
	- Should have at least two models to compare (?)
	- Again explain with CAM / saliency etc

* Link between regions and atlases
	- n regions -> n networks, then need to combine to reach a classification
	- Talaraich, thesis from Amira, nibabel?

# Meeting 15 (2021-04-12)
* AAL:
	- Verify that regions appear to be correct
	- Proceed to classify
		- Zeroing vs bounding-boxes
		- Summed intensity vs density from CAM

* ROI:
	- Non-maximum suppression
	- Regional analysis
		- What regions correspond to what condition, and how much of that region is covered etc

* Utexpo:
	- Encouraged, not required

# Meeting 16 (2021-04-26)
* AAL:
	- Some background picked up on CAM, use AAL to set these to zero (or just exclude background anyway)
	- Bar plot of CAM intensity per AAL region
	- Perform tests with CV where model is restricted to N top-ranked regions according to the CAM intensity
	- Also perform tests where model is restricted to higher-level regions (right/left, frontal, etc)
	- Analyse difference in CAM between correctly and incorrectly classified scans, in an attempt to explain the cause of incorrect classifications

# Meeting MIA (unknown)

# Meeting 17? (2021-05-12)
* Suggestions
	* Make all blue references to acronyms black?
		* Answer: Blue is fine

* Introduction
	* Remove concepts from Introduction, clearer workflow without introducing advanced concepts.

* Literature survey
	* Move CAM calculation to methodology
	* Literature survey are good to have a conclusion. Should be used to narrow down to what we try to solve
* Method
	* Method should only include approach but not results.
	* Introduce CNN then backbone network.
	* Move Table 4 to results (If we did it then move to result)
	* Include which evaluation metrics in Methodology
	* Add section about classification and features?
	* Rewrite overview (it is confusing that we split, into two parts. Make this clear). Features and methods
	* Move data out from methodology!
	* Implementation -> Bounding-boxes?

* Result
	* Add confusion matrix for BBOX. Confusing to relate between works.
	* Results: Logical progression, feature generation
*Discussion
	* Sustainability/Ethics, data or discussion section?
* Conclusion
	* Start Conclusion with why we tried to solve this problem
	* Conclusion should answer the AIM. Copy AIM and restate it. Bold or italic is good.

* Other
	* Unclear of sections: Bounding-box, anatomical and full-brain scan classifier
	* Tense's: Method: Introduction: Future, Literature survey: Past/Present, Method: Present, Result:Past, Discussion: Present, Conclusion: Combination
	* ADNI-1 is a acronym for ADNI.
	* Secure argumentation
	* Benefitial to our solution

# Meeting 17 (2021-05-12) (Eriks notes)
* Explain why other architectures of backbone network performed so poorly

* Add normalized y-axis to barplots

* Tense
	* Intro: Future
	* Literature: Past
	* Data: Present
	* Method: Present
	* Results: Past
	* Discussion: Mix
	* Conclusion: Mix

* Method
	* Massive reshuffling
	* Expand the 'overview' part a lot
	* Two feature generation sections:
		* Bbox
		* AAL
	* One classification section

* Extract data section before method

* Theory
	* More theory in method, how does a CNN even work etc
		* Then introduce specific implementations (ResNet, Vgg) after explanation of CNN
	* Move CAM mathematics from literature to method

* Literature
	* Conclusion should explain how we use previous work to limit and define our own work

* Intro
	* Don't need to include the entire solution, just the idea
	* Good three first paragraphs, refine last ones

* Naming consistency
	* Full-volume classifier -> Backbone network

* Result
	* Confusion matrix for Bbox
	* Barplots: add last line of captions to free-flowing text aswell

* Conclusion
	* Copy research questions and answer them one-by-one
		* Paragraph with research question in **bold** or *italic* above
	* Sustainability

* Attribution method
	* Not absolute truth

# Meeting 18 (2021-05-20)
* Intro
	- Remove dataset section

* Data
	- 'Explaration' unclear word, and maybe merge with preparation

* Method
	- Clearly define all variables used in equations
	- ROIAlign mentioned before explained
	- Introduce the AAL model clearly

* Results
	- Table with comparison to previous classification performances
		- What are they classifying? Binary?
		- What model are they using?
	- Select most important barplot and keep in results, move rest to appendix
		- Things that are purely 'systematic' can often be moved to appendix

* Discussion
	- Combine with results
		- 5.5 discussion. Discussion that cannot 'wait' can be presented immediately after result, but basically everything we have can 'wait'
	- Sustainability
		- Include all this in the introduction to the conclusion

* Conclusion
	- Introduce by restating the relevance of the project (sustainability)

# Meeting 19 (2021-06-01)
## Comments from supervisors
* Introduction: 
	* Add references of most important papers
	* Inclear what the difference is compared to state of the art.
	
	* Better to mention acronym BBox instead of Bounding-boxes - Fixed

* Contribution:
	* If not in introduction say here what have done before.
	

* Overview:
	* Clear for supervisors but unclear for people unknown wíthin the area
	* Sentence to say that we want to build a classifier and then apply BBox and AAL
	* Unlcear that we want to build a classifier
	
	* Unclear for listeners
		* Suggests animation for input and output
		* Example of each step with input/output.
		* Nice steps but not all people might understand so an example would be good.
		
		
* WSOL - Methodology:
	* Say that ResNet50 and VGG16 are more complicated than ResNet50 and VGG16. Unclear since we only show 3 architectures
	* Unclear why we use CAM. That it is done for explainability. Highlight that 
	* Why we use CAM over other explainability methods.
	* Just mention diffent techniques - Skip details - to get more time
	* Remove equation of CAM

* WSOL - Result: 
	* What is the reduce factor? Add stride
	* Unclear of images - Fixed
	
* BBOX - Methodology 
	* Better relate words to algorithm.
	* Move NMS and ROIAlign to a new slide. 
	
	
* BBOX - Results
	* Score not very informative
	* More interested of regions
	* Add score of BBox
	* Add arrow to bbox and show score, intensity and area
	* Idea: Zoom on one or two slices and display score
	
* AAL - Methodology:
	* Missing bar chart of ranked regions

* AAL - Result:
	* Table unclear and what is PAN and how what is the procedure to rank regions?
	* Lots of white space - Add bar chart to improve the explainability of regions?
	* Chart hard to read - Fixed
	
* Result
	* Unclear of improvement
		* High chance of comments to compare different architectures
		* Which architecture was used for BBox and AAL? 
		* How much reduction in term of area for BBox and AAL?
		* Time difference between Custom CNN and Simplified ResNet?
		* Measure the difference in performance of accuracy/scores of backbone vs regions-restricted.
	* How many regions did AAL use?

* Conclusion
	* Write complete sentences of specific context. Highlight the contribution with sentences
	* Choice of word;´Selling ourself a bit since we have good results
		* Needs to be more clear
	
	* Dont use questions without comments and better relate them to research questions
	
* Future work
	* Use more words/ a phrase so people connect speak with words.
	
