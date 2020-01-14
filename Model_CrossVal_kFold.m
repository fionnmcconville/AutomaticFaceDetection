clear all
close all

addpath .\SVM-KM\  %In order to use the provided SVM toolbox. We must add the path to the toolbox folder. 

rng default %Set random seed for reproducability of results


%% Extracting the data
%We should only change the sampling value of loadFaceImages if we are doing
%processing later that takes up a lot of time
[trainFeatures, trainLabs] = loadFaceImages('face_train.cdataset', 1);
[testFeatures, testLabs] = loadFaceImages('face_test.cdataset', 1);

%% Combining the available data - will be later split by CV
Features = [trainFeatures; testFeatures];
Labels = [trainLabs;testLabs];

%% Record original train features and test features, later will be used in CV_Wrong_frequency
train_Features = trainFeatures;
test_Features = testFeatures;

%% Sequence of original features
Sequence = (1:size(Labels,1));
Sequence = Sequence';

%% To record falsely classified features
Wrong_Record = int16.empty;

%% K-Fold Cross-Validaion
K = 5; %Specify the amount of folds you want here
indices = crossvalind('Kfold', Labels, K); %Function assigns each index in database a fold number between 1 and K 

for j = 1:K
    tic
    %Obtaining test and train data for this fold (the fold is i)
    testInds = (indices == j);
    trainInds = ~testInds;
    
    %Obtaining testing and training features
    trainFeatures = Features(trainInds,:);
    testFeatures = Features(testInds,:);
    
    %Obtaining testing and training labels
    trainLabs = Labels(trainInds,:);
    testLabs = Labels(testInds,:);
    
    %Obtaining testing features Sequence
    trainSequence = Sequence(trainInds,:);
    testSequence = Sequence(testInds,:);
    
%% Augmenting images so we multiply our dataset by 10. (We get 10 slightly different versions of image)
%We have turned off the augmentation option in the loadFaceImages dataset
%as we need to randomly each image, put them in their respective category
%(training or testing) and THEN we can augment the images. If we do this
%before we cross validate then we will get an over-optimistic accuracy
%returned on our model
[trainFeatures, trainLabs, trainSequence] = CV_augmentImages(trainFeatures, trainLabs, trainSequence); % For training

[testFeatures, testLabs, testSequence] = CV_augmentImages(testFeatures, testLabs, testSequence); %For testing

testSize = size(testFeatures, 1);
trainSize = size(trainFeatures, 1);
numFeatures = size(trainFeatures,2);


%% Pre-processing images
%Training images
for i = 1:trainSize
    Im = reshape(trainFeatures(i,:),27,18);
    
    %Im = enhanceContrastHE(uint8(Im));

    Im = enhanceContrastALS(uint8(Im));
    
    trainFeatures(i,:) = Im(:)';
   
end

%Testing Images 
for i = 1:testSize
    Im = reshape(testFeatures(i,:),27,18);
    
    %Im = enhanceContrastHE(uint8(Im));

    Im = enhanceContrastALS(uint8(Im));

    testFeatures(i,:) = Im(:)';
   
end

% %% Gabor Feature Descriptor 
% %Convert train & test features to images so they can be processed by gabor_features function
% gabortrainFeatures = zeros(trainSize, numFeatures * 40);
% for i = 1:trainSize
%     Im = reshape(trainFeatures(i,:),27,18);
%     
%     Im = gabor_feature_vector(uint8(Im)); %Produces 19,440 features 
%     gabortrainFeatures(i,:) = Im;
% end
% 
% gabortestFeatures = zeros(testSize, numFeatures * 40);
% for i = 1:testSize
%     Im = reshape(testFeatures(i,:),27,18);
%  
%         
%     Im = gabor_feature_vector(uint8(Im)); %Produces 19,440 features 
%     gabortestFeatures(i,:) = Im;
% end


%% PCA Feature Descriptor - Uncomment if using PCA Features
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(trainFeatures);

%% LDA
%[eigenVectors, eigenvalues, meanX, Xpca] = LDA(trainLabs,[], trainFeatures);


%% Training models - Full images
%Supervised Nearest Neighbour Training
%modelNN = NNtraining(trainFeatures, trainLabs);

%Supervised SVM Training
%modelSVM = SVMtraining(trainFeatures, trainLabs);


%% Training models - Gabor features
%modelNN = NNtraining(gabortrainFeatures, trainLabs);

%modelSVM = SVMtraining(gabortrainFeatures, trainLabs);


%% Training models - PCA
%modelNN = NNtraining(Xpca, trainLabs);

modelSVM = PCASVMtraining(Xpca, trainLabs, meanX, eigenVectors);



%% Testing model - SVM

classificationResult = [];%Re-initialise classificationResult vector - just in case of uneven split of fold size
for i =1:testSize 
       
    %% Full Image
%     testnumber= testFeatures(i,:);
%     classificationResult(i,1) = SVMTesting(testnumber,modelSVM);


    % Gabor Features
%     testnumber= gabortestFeatures(i,:); 
%     classificationResult(i,1) = SVMTesting(testnumber,modelSVM); 
%     

    %% PCA Features
    testnumber= testFeatures(i,:);
    Xpca = (testnumber - meanX) * eigenVectors; 
    classificationResult(i,1) = SVMTesting(Xpca,modelSVM); 
    
    
end


%% Evaluation - Recognition Rate

% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the testing image
comparison = (testLabs==classificationResult);

%Accuracy is the most common metric. It is defiend as the number of
%correctly classified samples/ the total number of tested samples
Accuracy = sum(comparison)/length(comparison);

%% Evaluation - TP,FP,TN,FN

%True Positive: Sum of Face images predicted correctly
%False Positive: Sum of Non-Face images predicted wrongly
%True Negative: Sum of Non-Face images predicted correctly
%False Negative: Sum of Face images predicted  wrongly
[TP, FP, TN, FN] = TP_FP_TN_FN(testLabs, classificationResult);

%% Evaluation -  Precision, recall, specificity, etc
%It is based on value of TP, FP, TN, FN
[Recall, Precision, Specificity, Sensitivity, F_measure, False_alarm_rate]  = Precision_Sensitivity(TP, FP, TN, FN);

%% Record evaluation data 
accuracy(1,j) = Accuracy; 
tp(1,j) = TP;
fp(1,j) = FP;
tn(1,j) = TN;
fn(1,j) = FN;
recall(1,j) = Recall;
precision(1,j) = Precision;
sensitivity(1,j) = Sensitivity;
specificity(1,j) = Specificity;
f_measure(1,j) = F_measure;
false_alarm_rate(1,j) = False_alarm_rate;

%Record  Wrong Classified features
i=1;
while i<=length(comparison) 
    if ~comparison(i)
        Wrong_Record = [Wrong_Record; testSequence(i,:)];
    end
    i=i+1;   
end
toc
end 


%% Show Images that were most often mislclassified. Could give an idea of the strengths and weaknesses of a particular model
[Sort_Wrong_Record] = CV_Wrong_Frequency(Wrong_Record, train_Features, test_Features);

%% Cross Validated Average of each Evaluation element and output it in a table
[Accuracy] = CV_Evaluation(trainLabs, testLabs, accuracy, tp, fp, tn, fn, recall, precision, sensitivity, specificity, f_measure, false_alarm_rate, K);

