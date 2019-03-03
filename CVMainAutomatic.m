clear all
close all

addpath .\SVM-KM\ 
%In order to use the provided SVM toolbox. We must add the path to the toolbox folder. 
%This folder is in prac 5


%% Easiest and most basic method of feature extraction and sampling for training and testing data.
%Might be useful to compare accuracy of basic method with attempts of
%improving later. For the moment this method of sampling is better because the
%loadFaceImages method has an inbuilt data augmentation functionality which
%uses various methods to increase the size of the dataset. Maybe later we
%will learn these methods in the course and be able to use them in our
%manual setup. Maybe the manual setup isn't necessary. For now though this gives the optimum accuracy

%% First extracting training images
[trainFeatures, trainLabs] = loadFaceImages('face_train.cdataset', 1);
[testFeatures, testLabs] = loadFaceImages('face_test.cdataset', 1);
%Returns a training data size of 670 images from the 94 images in the
%training dataset.

%Half/half way to divide train and test dataset in average
%[trainFeatures, testFeatures] = Halfhalf('face_train.cdataset', 'face_test.cdataset');


%% Note:
%We should only change the sampling value of loadFaceImages if we are doing
%processing later that takes up a lot of time

%% Gabor features - Training feature descriptor
%Converting trainfeatures data to images where they can be preprocessed and
%also gabor features function can be implemented
gabortrainFeatures = zeros(670, 19440);

for i = 1:size(trainFeatures, 1)
    Im_train = reshape(trainFeatures(i,:),27,18);
    
    Im_train = enhanceContrastHE(uint8(Im_train));
    %Hist Equalisation. Gives us a large improvement when used with gabor.
    %Accuracy is 0.9042
    
    %Im = enhanceContrastALS(uint8(Im));
    %Automatic Linear stretching. Gives us an improvement when used with gabor.
    %Accuracy is 0.8292
        
    Im_train = gabor_feature_vector(Im_train); %Produces 19,440 features 
    gabortrainFeatures(i,:) = Im_train;
end

%% Gabor features - testing feature descriptor
%Converting trainfeatures data to images where they can be preprocessed and
%also gabor features function can be implemented
gabortestFeatures = zeros(670, 19440);

for i = 1:size(testFeatures, 1)
 
    Im_test = reshape(testFeatures(i,:),27,18);
    
    Im_test = enhanceContrastHE(uint8(Im_test));
    %Hist Equalisation. Gives us a large improvement when used with gabor.
    %Accuracy is 0.9042
    
    %Im = enhanceContrastALS(uint8(Im));
    %Automatic Linear stretching. Gives us an improvement when used with gabor.
    %Accuracy is 0.8292
        
    Im_test = gabor_feature_vector(Im_test); %Produces 19,440 features 
    gabortestFeatures(i,:) = Im_test;
end

%%Combine train features and test features for cross-validation model
gaborFeatures = gabortrainFeatures + gabortestFeatures;
%% Training models - Full images
%Supervised Nearest Neighbour Training
%modelNN = NNtraining(trainFeatures, trainLabs);

%Supervised SVM Training
%modelSVM = SVMtraining(trainFeatures, trainLabs);

%% Training models - Gabor features
%modelNN = NNtraining(gabortrainFeatures, trainLabs);


%1, Cross-validation model with SVM
%[testFeatures, testLabs] = loadFaceImages('face_test.cdataset', 1);
%Features = trainFeatures;
%modelSVM = SVMtraining(gaborFeatures, Labs);

%Using cross-validation to build the classification model
%svmmodel = crossval(modelSVM);

%Find the cross-validated loss of classifier
%loss = kfoldLoss(modelsvm);


%2, Cross-validation model with k-nearest neighbor
%[testFeatures, testLabs] = loadFaceImages('face_test.cdataset', 1);
%Features = trainFeatures;
modelKNN = CVknn(gaborFeatures, trainLabs);


%Find the cross-validated loss of classifier
loss = kfoldLoss(modelKNN);

%Estimate accuracy of model
Accuracy = 1 - loss;
disp(Accuracy);
