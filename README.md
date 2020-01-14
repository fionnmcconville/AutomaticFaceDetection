# Automatic Face Detection Using Sliding Window Detector
Creating an automatic face detector using matlab and machine learning. In particular KNN and SVM with a variety of kernel functions are used. Also, deep analysis and evaluation are implemented upon the results.

Optimal parameters have been found using a bayes search approach for each kernel function and have been already set up.

### To Test Type of Evaluation:
Run Model_CrossVal_kFold
or
Run Model_LeavePOut

### To Create a final model using all of the data to train:
Run FinalModel

### To Run the sliding window detector:
Run Detector

When Choosing a particular feature type to use - choose either Gabor features or PCA, make sure to comment out relevant bits of the code. 

For Example, if evaluating a model with gabor features, make sure to comment out all sections which have been marked as a PCA relevant. These sections are clearly commented and easy to read.

When runnning the final detector, make sure all of the code in the SlidingWindow function is correctly aligned with the feature descriptor you have used.

If you wish to view just an end summary of results, then run the script Result_Summary. This script loads a saved workspace with all of the detections pre-loaded from each model. Simply run the script to view the results.


