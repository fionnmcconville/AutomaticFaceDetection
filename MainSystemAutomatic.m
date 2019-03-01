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
%Returns a training data size of 670 images from the 94 images in the
%training dataset.

%% Note:
%We should only change the sampling value of loadFaceImages if we are doing
%processing later that takes up a lot of time

%% Gabor features - Training feature descriptor
%Converting trainfeatures data to images where they can be preprocessed and
%also gabor features function can be implemented
gabortrainFeatures = zeros(670, 19440);
for i = 1:size(trainFeatures, 1)
    Im = reshape(trainFeatures(i,:),27,18);
    
    Im = enhanceContrastHE(uint8(Im));
    %Hist Equalisation. Gives us a large improvement when used with gabor.
    %Accuracy is 0.9042
    
    %Im = enhanceContrastALS(uint8(Im));
    %Automatic Linear stretching. Gives us an improvement when used with gabor.
    %Accuracy is 0.8292
        
    Im = gabor_feature_vector(Im); %Produces 19,440 features 
    gabortrainFeatures(i,:) = Im;
end
%% Training models - Full images
%Supervised Nearest Neighbour Training
%modelNN = NNtraining(trainFeatures, trainLabs);

%Supervised SVM Training
%modelSVM = SVMtraining(trainFeatures, trainLabs);

%% Training models - Gabor features
%modelNN = NNtraining(gabortrainFeatures, trainLabs);

%1, Supervised SVM Training(devided into train data and test data)
modelSVM = SVMtraining(gabortrainFeatures, trainLabs);

%2, Building cross-validation model with SVM
%Features = trainFeatures + testFeatures;
%modelSVM = CVsvm(Features, trainLabs);

%Find the cross-validated loss of classifier
%loss = kfoldLoss(modelSVM);

%Estimate accuracy of model
%Accuracy = 1 - loss;

%3, Building cross-validation model with k-nearest neighbor
%Features = trainFeatures + testFeatures;
%modelKNN = CVknn(Features, trainLabs);

%Find the cross-validated loss of classifier
%loss = kfoldLoss(modelKNN);

%Estimate accuracy of model
%Accuracy = 1 - loss;

%% Then extracting testing images
[testFeatures, testLabs] = loadFaceImages('face_test.cdataset', 1);
%Returns a testing data size of 240 from the 30 images in the
%testing dataset

%% Gabor features - Testing feature descriptor - Comment out if using another feature method
%Converting trainfeatures data to images where they can be preprocessed and
%also gabor features function can be implemented
gabortestFeatures = zeros(240, 19440);
for i = 1:size(testFeatures, 1)
    Im = reshape(testFeatures(i,:),27,18);
    
    Im = enhanceContrastHE(uint8(Im));
    %Hist Equalisation. For the moment it doesn't give us improved results
    
    %Im = enhanceContrastALS(uint8(Im));
    %Automatic Linear stretching. For the moment it doesn't give us improved results
        
    Im = gabor_feature_vector(Im); %Produces 19,440 features 
    gabortestFeatures(i,:) = Im;
end

%% Testing model
for i=1:size(testFeatures,1)
    
    %testnumber= testFeatures(i,:); % For full image feature descriptor
    
    testnumber= gabortestFeatures(i,:); % For gabor feature descriptor

    %% NN model
    %classificationResult(i,1) = NNTesting(testnumber, modelNN);
    %Accuracy is 0.5083. Which is again strange. Gives a reverse of what
    %happened in the manual system. With k = 1 in the KNNTesting the
    %accuracy is around 0.7
    
    %% KNN model
    %classificationResult(i,1) = KNNTesting(testnumber, modelNN, 4);
    
    %% SVM Model
    classificationResult(i,1) = SVMTesting(testnumber,modelSVM);
    %SVM accuracy with no changed parameters and full image feature desc is 0.7083
    %SVM accuracy with no changed parameters, hist equalisation
    %and gabor feature desc is 0.9042

end

%% This bit is just to record a table for all of the accuracies for varying amounts of K in KNN
%Could be useful for the report later

% accuracyForEachK(20,2) = 0;
% accuracyForEachK(:,1) = 1:20;
% for k = 1:20
%    
%     for i = 1:size(testFeatures,1)
%         testnumber= testFeatures(i,:);
%         classificationResult(i,1) = KNNTesting(testnumber, modelNN, k);
%     end 
%     
%     comparison = (testLabs==classificationResult);
%     Accuracy = sum(comparison)/length(comparison);
%     accuracyForEachK(k,2) = Accuracy; 
%     %accuracyForEachK is the final table holding the accuracy of the model
%     %for each value of K from 1:20
% end
% 
% 
% accuracyForEachK

%% Evaluation

% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the testing image
comparison = (testLabs==classificationResult);

%Accuracy is the most common metric. It is defiend as the number of
%correctly classified samples/ the total number of tested samples
Accuracy = sum(comparison)/length(comparison)


%We display all of the correctly classified images. (Max is around 25)
figure, 
sgtitle('Correct Classification'),
colormap(gray)
count=0;
i=1;
while (count<25)&&(i<=length(comparison))
   
    if comparison(i)
        count=count+1;
        subplot(5,5,count)
        Im = reshape(testFeatures(i,:),27,18);
        imagesc(Im)
        axis off
    end
    
    i=i+1;
    
end


%We display all of the incorrectly classified images. (Max is around 25)
figure
sgtitle('Wrong Classification'),
colormap(gray)
count=0;
i=1;
while (count<25)&&(i<=length(comparison))
    
    if ~comparison(i)
        count=count+1;
        subplot(5,5,count)
        Im = reshape(testFeatures(i,:),27,18);
        imagesc(Im)
        title(labelToDescription(classificationResult(i)))
        axis off
    end
    
    i=i+1;
    
end
