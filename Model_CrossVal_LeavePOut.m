%% Model for leave-p out Cross-Validation
%%With SVM training and testing, gaborfeatures
%%With Augmentation
%%With Accuracy, TP/FP/TN/FN, recall, sentivity and etc.

clear all
close all

addpath .\SVM-KM\ 
%In order to use the provided SVM toolbox. We must add the path to the toolbox folder. 
%This folder is in prac 5

rng default %Set random seed for reproducability of results

%% We extract both databases and pass them through a sampling method

%We should only change the sampling value of loadFaceImages if we are doing
%processing later that takes up a lot of time
[trainFeatures, trainLabs] = loadFaceImages('face_train.cdataset', 1);
[testFeatures, testLabs] = loadFaceImages('face_test.cdataset', 1);

%% Initialise Evaluation elements record Cross-validation for Each Features Set
accuracy = [];
tp = [];
fp = [];
tn = [];
fn = [];
recall = [];
precision = [];
specificity = [];
sensitivity = [];
f_measure = [];
false_alarm_rate = [];

%% For Recording wrongly all classified features 
Wrong_Record = int16.empty;

%% Size of original training and testing features 
size_trainLabs = size(trainLabs);
size_testLabs = size(testLabs);

%% Sequence of original Features (order)
trainSequence = (1:size_trainLabs(1,1));
testSequence = (size_trainLabs(1,1)+1:size_testLabs(1,1) + size_trainLabs(1,1));
trainSequence = trainSequence';
testSequence = testSequence';


%% Total label length of trainFeatures and testFeatures
length = size(trainLabs) + size(testLabs);

%% n equal the number of folds for Cross-valuidation
%Put one element in each fold to build testing features each time
%26 to ensure 20% testing features
n = 26; 

%% Number of features in each fold
if(mod(length(1,1),n) ~= 0)
    fold_size = int16(length(1,1) / (n - 1) - mod(length(1,1) / (n - 1), 1));
    if fold_size * n < length(1,1)
        fold_size = fold_size + 1;
    end
else
    fold_size = int16(length(1,1) / n);
end

%% Build models
% number of models equal to size of a fold
for f = 1:fold_size(1,1)

tic
%% New training & testing features for building models
trainFeatures_fold = trainFeatures;
trainLabs_fold = trainLabs;
testFeatures_fold = testFeatures;
testLabs_fold = testLabs;
trainSequence_fold = trainSequence;
testSequence_fold = testSequence;

%% Re-classify features 

[trainFeatures_fold, testFeatures_fold, trainLabs_fold, testLabs_fold, trainSequence_fold, testSequence_fold] = CV_LeavePOut_func(trainFeatures_fold, testFeatures_fold, trainLabs_fold, testLabs_fold, trainSequence_fold,testSequence_fold, n, fold_size(1,1), length, f);

%% Augmentation only for Cross-validation
[trainFeatures_fold, trainLabs_fold, trainSequence_fold] = CV_augmentImages(trainFeatures_fold, trainLabs_fold, trainSequence_fold); % For training
[testFeatures_fold, testLabs_fold, testSequence_fold] = CV_augmentImages(testFeatures_fold, testLabs_fold, testSequence_fold); %For testing

testSize = size(testFeatures_fold, 1);
trainSize = size(trainFeatures_fold, 1);
numFeatures = size(trainFeatures_fold,2);

%% Pre-processing images
%Training images
for i = 1:trainSize
    Im = reshape(trainFeatures_fold(i,:),27,18);
    
    %Im = enhanceContrastHE(uint8(Im));

    Im = enhanceContrastALS(uint8(Im));
    
    trainFeatures_fold(i,:) = Im(:)';
   
end

%Testing Images 
for i = 1:testSize
    Im = reshape(testFeatures_fold(i,:),27,18);
    
    %Im = enhanceContrastHE(uint8(Im));

    Im = enhanceContrastALS(uint8(Im));

    testFeatures_fold(i,:) = Im(:)';
   
end

%% Gabor Feature Descriptor
%Convert train & test features to images so they can be processed by gabor_features function
gabortrainFeatures = zeros(trainSize, numFeatures * 40);
for i = 1:trainSize
    Im = reshape(trainFeatures_fold(i,:),27,18);
    
    Im = gabor_feature_vector(uint8(Im)); %Produces 19,440 features 
    gabortrainFeatures(i,:) = Im;
end

gabortestFeatures = zeros(testSize, numFeatures * 40);
for i = 1:testSize
    Im = reshape(testFeatures_fold(i,:),27,18);
 
        
    Im = gabor_feature_vector(uint8(Im)); %Produces 19,440 features 
    gabortestFeatures(i,:) = Im;
end

% %% PCA Feature Descriptor
% [eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(trainFeatures_fold);


%% Training models - Full images
%Supervised Nearest Neighbour Training
%modelNN = NNtraining(trainFeatures_fold, trainLabs_fold);

%Supervised SVM Training
%modelSVM = SVMtraining(trainFeatures_fold, trainLabs_fold);


%% Training models - Gabor features
%modelNN = NNtraining(gabortrainFeatures, trainLabs_fold);

modelSVM = SVMtraining(gabortrainFeatures, trainLabs_fold);


%% Training models - PCA
%modelNN = NNtraining(Xpca, trainLabs_fold);

%modelSVM = PCASVMtraining(Xpca, trainLabs_fold, meanX, eigenVectors);


%% Reinitialise classificationResult
classificationResult = [];
%% Testing model
for i=1:testSize 
    
       %% Full Image
%     testnumber= testFeatures_fold(i,:);
%     classificationResult(i,1) = SVMTesting(testnumber,modelSVM);
%     
    %% Gabor Features
    testnumber= gabortestFeatures(i,:); % For gabor feature descriptor
    classificationResult(i,1) = SVMTesting(testnumber,modelSVM); %Gabor
     
    %% PCA Features
%     testnumber= testFeatures_fold(i,:);
%     Xpca = (testnumber - meanX) * eigenVectors; %For PCA
%     classificationResult(i,1) = SVMTesting(Xpca,modelSVM); %PCA

end

%% Evaluation - Accuracy
% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the testing image
comparison = (testLabs_fold==classificationResult);
%Accuracy is the most common metric. It is defiend as the number of
%correctly classified samples/ the total number of tested samples
comparison_size = size(comparison);
Accuracy = sum(comparison)/comparison_size(1,1);

%% Record Wrong Classified features
i=1;
while i<=comparison_size(1,1)   
    if ~comparison(i)
        Wrong_Record = [Wrong_Record; testSequence_fold(i,:)];
    end
    i=i+1;   
end



%% Evaluation - TP,FP,TN,FN

%True Positive: Sum of Face images predicted correctly
%False Positive: Sum of Non-Face images predicted wrongly
%True Negative: Sum of Non-Face images predicted correctly
%False Negative: Sum of Face images predicted  wrongly
[TP, FP, TN, FN] = TP_FP_TN_FN(testLabs_fold, classificationResult);

%% Evaluation -  Precision, recall, specificity, etc
%It is based on value of TP, FP, TN, FN
[Recall, Precision, Specificity, Sensitivity, F_measure, False_alarm_rate]  = Precision_Sensitivity(TP, FP, TN, FN);

%% Record evaluation data 
accuracy = [accuracy, Accuracy];
tp = [tp,TP];
fp = [fp, FP];
tn = [tn, TN];
fn = [fn, FN];
recall = [recall, Recall];
precision = [precision, Precision];
sensitivity = [sensitivity, Sensitivity];
specificity = [specificity, Specificity];
f_measure = [f_measure, F_measure];
false_alarm_rate = [false_alarm_rate, False_alarm_rate];

toc
end

tic
%% Record Features with highest Frequency to be falsely classified
[Sort_Wrong_Record] = CV_Wrong_Frequency(Wrong_Record, trainFeatures, testFeatures);

%% Average of each Evaluation element and output it in figures
[Accuracy] = CV_Evaluation(trainLabs_fold, testLabs_fold, accuracy, tp, fp, tn, fn, recall, precision, sensitivity, specificity, f_measure, false_alarm_rate, fold_size);


