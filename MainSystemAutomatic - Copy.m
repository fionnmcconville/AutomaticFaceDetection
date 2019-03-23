clear all
close all

addpath .\SVM-KM\ 
%In order to use the provided SVM toolbox. We must add the path to the toolbox folder. 
%This folder is in prac 5


%% We extract both databases and pass them through a sampling method
% (either half/half or cross validation) in order to get a random sample

%We should only change the sampling value of loadFaceImages if we are doing
%processing later that takes up a lot of time
[trainFeatures, trainLabs] = loadFaceImages('face_train.cdataset', 1);
[testFeatures, testLabs] = loadFaceImages('face_test.cdataset', 1);



%% 1: Half/half way to divide train and test dataset equally
%[trainFeatures, testFeatures,trainLabs, testLabs] = Halfhalf(trainFeatures, testFeatures, trainLabs, testLabs);


%% 2: Cross-validation way to classify the features
%This is a manual way to execute cross-validation. The newest test features are selected out randomly
%in every fold. Function chooses test sample in every fold randomly
n = 20; %n equal the fold number
[trainFeatures, testFeatures, trainLabs, testLabs] = CrossValidation(trainFeatures, testFeatures, trainLabs, testLabs, n);


%% Augmenting images so we multiply our dataset by 10. (We get 10 slightly different versions of image)
%We have turned off the augmentation option in the loadFaceImages dataset
%as we need to randomly each image, put them in their respective category
%(training or testing) and THEN we can augment the images. If we do this
%before we cross validate then we will get an over-optimistic accuracy
%returned on our model
[trainFeatures, trainLabs] = augmentImages(trainFeatures, trainLabs); % For training

[testFeatures, testLabs] = augmentImages(testFeatures, testLabs); %For testing

testSize = size(testFeatures, 1);
trainSize = size(trainFeatures, 1);
numFeatures = size(trainFeatures,2);

%% Lets check if augmentation worked ok
% figure,
% colormap(gray),
% sgtitle("Displaying the first 10 augmented training images"),
% for i=1:10
% 
%     Im = squeeze(trainFeatures(i,:,:)); 
%     I = reshape(Im,27,18);
%     subplot(2,5,i), imagesc(I), title(['label: ',labelToDescription(trainLabs(i,1))])
%     axis off
% end

%% Gabor features - Training feature descriptor
%Converting trainfeatures data to images where they can be preprocessed and
%also gabor features function can be implemented

gabortrainFeatures = zeros(trainSize, numFeatures * 40);
for i = 1:trainSize
    Im = reshape(trainFeatures(i,:),27,18);
    
    %Im = enhanceContrastHE(uint8(Im));
    %Hist Equalisation. Gives us a large improvement when used with gabor.
    %Accuracy is 0.92 - 0.93
    
    Im = enhanceContrastALS(uint8(Im));
    %Automatic Linear stretching. Gives us an improvement when used with gabor.
    %Accuracy is ~ 0.95
        
    Im = gabor_feature_vector(uint8(Im)); %Produces 19,440 features 
    gabortrainFeatures(i,:) = Im;
end

%% Gabor features - Testing feature descriptor - Comment out if using another feature method
%Converting trainfeatures data to images where they can be preprocessed and
%also gabor features function can be implemented
gabortestFeatures = zeros(testSize, numFeatures * 40);
for i = 1:testSize
    Im = reshape(testFeatures(i,:),27,18);
    
    %Im = enhanceContrastHE(uint8(Im));
    %Hist Equalisation. 
    
    Im = enhanceContrastALS(uint8(Im));
    %Automatic Linear stretching. 
        
    Im = gabor_feature_vector(uint8(Im)); %Produces 19,440 features 
    gabortestFeatures(i,:) = Im;
end



%% Training models - Full images
%Supervised Nearest Neighbour Training
%modelNN = NNtraining(trainFeatures, trainLabs);

%Supervised SVM Training
%modelSVM = SVMtraining(trainFeatures, trainLabs);

%% Training models - Gabor features
%modelNN = NNtraining(gabortrainFeatures, trainLabs);

% Supervised SVM Training(devided into train data and test data)
modelSVM = SVMtraining(gabortrainFeatures, trainLabs);
save detectorModelSVM modelSVM

%% Testing model

for i=1:testSize 
    
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
%     %for i = 1:size(testFeatures,1)
%         %testnumber= testFeatures(i,:); %For full image
%      for i = 1:size(gabortestFeatures,1)
%         testnumber= gabortestFeatures(i,:); % For gabor feature descriptor
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
%     %Show accuracy table out in a figure
% f = uifigure;
% uitable(f, 'Data', accuracyForEachK);



%% Evaluation - Accuracy

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


%% Evaluation - TP,FP,TN,FN

%True Positive: Sum of Face images predicted correctly
%False Positive: Sum of Non-Face images predicted wrongly
%True Negative: Sum of Non-Face images predicted correctly
%False Negative: Sum of Face images predicted  wrongly
[TP, FP, TN, FN] = TP_FP_TN_FN(testLabs, classificationResult);

%Build a confusion matrix
Confusion_Matrix = {'Actual_Face'; 'Actual_NonFace'};
Predict_Face = {TP; FP};
Predict_NonFace =  {FN; TN};
Confusion_Table = table(Confusion_Matrix, Predict_Face, Predict_NonFace);

%how confusion matrix in a figure
Confusion_matrix_show = uifigure;
uitable(Confusion_matrix_show, 'Data', Confusion_Table);
%% Evaluation -  Precision, recall, specificity, etc

%It is based on value of TP, FP, TN, FN
[Recall, Precision, Specificity, Sensitivity, F_measure, False_alarm_rate]  = Precision_Sensitivity(TP, FP, TN, FN);

%Record test images size and precision, recall, specificity, etc in a matrix
Evaluation_Name = {'Test images size'; 'Recall'; 'Precision'; 'Specificity';'Sensitivity'; 'F-measure'; 'False alarm rate'};
Values = {length(testLabs); Recall; Precision; Specificity; Sensitivity; F_measure; False_alarm_rate};
Precision_Sensitivity = table(Evaluation_Name, Values);

%Show test images size and precision, recall, specificity, etc in a figure
Precision_Sensitivity_show = uifigure;
uitable(Precision_Sensitivity_show, 'Data', Precision_Sensitivity);


%% Detection Implementation
im1 = imread('im1.jpg');
im2 = imread('im2.jpg');
im3 = imread('im3.jpg');
im4 = imread('im4.jpg');

detections = SlidingWindow(im3);
DrawBoundingBoxes(im3, detections);

% Non maxima Supression
detections = simpleNMS(detections);
DrawBoundingBoxes(im3, detections);


%% Evaluation
%Groundtruth
%im1 = 7 faces
%im2 = 15 faces
%im3 = 8 faces
%im4 = 57?
solutionTruth = [];
             
comparison = (predictedFullImage==solutionTruth);
Accuracy = sum(sum(comparison))/ (size(comparison,1)*size(comparison,2))