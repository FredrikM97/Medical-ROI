# Questions (20201109)
1. Difference between ROI and feature selection? Still feel like they are correlated but not the same. Reasoning: ROI could reduce dimension space and same with feature selection but is the idea that ROI also point out where and what is in the remaining dimension.

2. Part of the idea behind ROI to improve performance in speed of evaluating images?

3. Based on the description of project that ROI be considered as structure biomarkers. Does this mean that by implementing *INSERT solution* it could be possible to point out the most common areas for a selected target and then only train/test on these areas without losing relevant information. 

What I'm trying to say is: Asking ROI for positions of possible target areas and test classification on these areas and hopefully improve the accuracy by removing data that does not help the classification.

4. Reasoning number 1:
I think that ROI try to pinpoint position of target label or detects surfaces that stand out and (maybe?) removes the rest of the regions. This idea is based on Informative and Uninformative Regions Detection inWCE Frames [3].
- This sounds like anomaly detection? 
- Meanwhile feature selection try to reduce the number of features that contribute to the result 

5. Reasoning number 2:
From a report "Feature and Region Selection for Visual Learning"[1] they explain the idea of discriminative regions in images. So instead of selecting an point in the space or a "feature" we could select the regions that is most descrimative. This can be followed up by the explaination of descriminative regions by: "Recognition of similar characters using gradient features of discriminative regions"[2]. 



[1][http://dx.doi.org/10.1109/TIP.2016.2514503]
[2][https://doi.org/10.1016/j.eswa.2019.05.050]
[3][10.7726/jac.2014.1002a]

# Questions (20201116)
* Limitations of other projects? PCA? Reduced information of images
* Many reports mentions “center”. What does it mean? Accordingly, we recognize that our correction ofthe analyses for center might have resulted in a relative reduction in the sensitivity (but not in the specificity) of our analyses
* Concerns in how to make it novelty (Covers ROI and feature-map activation: Quantifying brain metabolism from FDG-PET)

* Comparison between feature-map activation and Supervised Approach and none region selection method

* Idea: Create embedding that utilites ROI or implement it directly on the CNN
* Still no response regarding dataset

# Questions (20201123)
* Are we using PET or PET/CT which stand for computed tomography?
	* Is it CT that creates the 2D slices which is then stacked upon each other?
	* https://www.nibib.nih.gov/science-education/science-topics/computed-tomography-ct
	* https://www.hopkinsmedicine.org/health/treatment-tests-and-therapies/positron-emission-tomography-pet
	* New technology to replace PET is The gamma camera system which is a faster and cheaper alternative 
		* According to https://www.sciencedirect.com/topics/medicine-and-dentistry/gamma-camera gamma camera is most common. Combine 

* Do we want to exclude layers or slices before sending them as input to NN?
* What are our limitations?
* What week is half-time seminar (deadlines?)