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



%Decide which Classification way to choose
% "1" means normal dividing into training set and testing set
% "2" means Half/half
% "3" means Cross-calidation(Recommended)
Which_Classification = 3



if Which_Classification == 2
%Half/half way to divide train and test Features equally
%Percentage is the occupation of test Features in all Features
%0.5 is default(Half/half), and you can modify it to be another value
percentage = 0.5;
[trainFeatures, testFeatures, trainLabs, testLabs] = Halfhalf(trainFeatures, testFeatures, trainingLabs, testingLabs, percentage);
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
<<<<<<< Updated upstream
=======
save detectorModelSVM modelSVM

>>>>>>> Stashed changes
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
end



if Which_Classification == 3
%Cross-validation: For each classification: Divide Features into n folds, and select one of features
%in each fold as test features, perform classification fold times with
%each classification's test features location different from each other
%.And then count the average accuracy of each classification's model.

%Evaluation variables record Cross-validation for Each Features Set
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

%Total label length of trainFeatures and testFeatures
length = size(trainLabs) + size(testLabs);

%n equal the number of fold 
n = 200; 

%Number of features in each fold
%Consider the situation that remainder exists
if(mod(length(1,1),n) ~= 0)
    fold_size = int16(length(1,1) / (n - 1) - mod(length(1,1) / (n - 1), 1));
    if fold_size * n < length(1,1)
        fold_size = fold_size + 1;
    end
else
    fold_size = int16(length(1,1) / n);
end

%Create matrix to record each Features set for Cross-validation
trainFeatures_matrix = {};
testFeatures_matrix = {};
trainLabs_matrix = {};
testLabs_matrix = {};

%Modeling tranining and testing, evaluation just for Cross-validation
for f = 1:fold_size(1,1)

if f > 1
    testFeatures_fold_size_before = size(testFeatures_fold);
end

%training and testing features and Labels when in number f Features Set
trainFeatures_fold = trainFeatures;
trainLabs_fold = trainLabs;
testFeatures_fold = testFeatures;
testLabs_fold = testLabs;

[trainFeatures_fold, testFeatures_fold, trainLabs_fold, testLabs_fold] = CrossValidation(trainFeatures_fold, testFeatures_fold, trainLabs_fold, testLabs_fold, n, fold_size(1,1), length, f);

%To make size of each column of train and test Features set is equal
%To prevent the last bag do not include a test features, a repeat features is allowed
testFeatures_fold_size_after = size(testFeatures_fold);
if f > 1 
    if testFeatures_fold_size_before(1,1) > testFeatures_fold_size_after(1,1)
       testFeatures_fold = [testFeatures_fold; testFeatures(testFeatures_fold_size_before(1,1),:)];
       testLabs_fold = [testLabs_fold; testLabs(testFeatures_fold_size_before(1,1),:)];
    else
       trainFeatures_fold = [trainFeatures_fold; testFeatures(testFeatures_fold_size_before(1,1),:)];
       trainLabs_fold = [trainLabs_fold; testLabs(testFeatures_fold_size_before(1,1),:)];
    end
end

[trainFeatures_fold, trainLabs_fold] = augmentImages(trainFeatures_fold, trainLabs_fold); % For training
[testFeatures_fold, testLabs_fold] = augmentImages(testFeatures_fold, testLabs_fold); %For testing

trainFeatures_matrix = [trainFeatures_matrix, trainFeatures_fold];
testFeatures_matrix = [testFeatures_matrix, testFeatures_fold];
trainLabs_matrix = [trainLabs_matrix, trainLabs_fold];
testLabs_matrix = [testLabs_matrix, testLabs_fold];

testSize = size(testFeatures_fold, 1);
trainSize = size(trainFeatures_fold, 1);
numFeatures = size(trainFeatures_fold,2);
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
    Im = reshape(trainFeatures_fold(i,:),27,18);
    
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
    Im = reshape(testFeatures_fold(i,:),27,18);
    
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
modelSVM = SVMtraining(trainFeatures, trainLabs);
%% Training models - Gabor features
%modelNN = NNtraining(gabortrainFeatures, trainLabs_fold);
% Supervised SVM Training(devided into train data and test data)
%modelSVM = SVMtraining(gabortrainFeatures, trainLabs);
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
comparison = (testLabs_fold==classificationResult);
%Accuracy is the most common metric. It is defiend as the number of
%correctly classified samples/ the total number of tested samples
comparison_size = size(comparison);
Accuracy = sum(comparison)/comparison_size(1,1)

%% We display at most 25 of the correctly classified images
%figure, 
%sgtitle('Correct Classification'),
%colormap(gray)
%count=0;
%i=1;
%while (count<25)&&(i<=comparison_size(1,1))
   
 %   if comparison(i)
  %      count=count+1;
   %     subplot(5,5,count)
    %    Im = reshape(testingFeatures(i,:),27,18);
     %   imagesc(Im)
      %  axis off
    %end
    
    %i=i+1;
    
%end


%We display all of the incorrectly classified images. (Max is around 25)
%figure
%sgtitle('Wrong Classification'),
%colormap(gray)
%count=0;
%i=1;
%while (count<25)&&(i<=length(comparison))
    
 %   if ~comparison(i)
  %      count=count+1;
   %     subplot(5,5,count)
    %    Im = reshape(testFeatures(i,:),27,18);
     %   imagesc(Im)
      %  title(labelToDescription(classificationResult(i)))
       % axis off
    %end
    
    %i=i+1;
    
%end

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

end

Recall = 0;
Precision = 0;
Specificity = 0;
Sensitivity = 0; 
F_measure = 0;
False_alarm_rate = 0;

% Table elements record accuracy in every Features sets(with different number
%of f)
CV_Sequence = {};
CV_Accuracy =  {};
Accuracy = 0;


%Record number of useful Precision, recall, specificity, etc
count_recall = 0;
count_precision = 0;
count_sensitivity = 0;
count_specificity = 0;
count_f_measure = 0;
count_false_alarm_rate = 0;

for j = 1:fold_size(1,1)
    Accuracy = Accuracy + accuracy(:,j);
    CV_Sequence = [CV_Sequence; j];
    CV_Accuracy = [CV_Accuracy; accuracy(:,j)];
    TP = TP + tp(:,j);
    FP = FP + fp(:,j);
    TN = TN + tn(:,j);
    FN = FN + fn(:,j);
    if recall(:,j) ~= Inf
        Recall = Recall + recall(:,j);
        count_recall = count_recall + 1;
    end
    if precision(:,j) ~= Inf
        Precision = Precision + precision(:,j);
        count_precision = count_precision + 1;
    end
    if sensitivity(:,j) ~= Inf
        Sensitivity = Sensitivity + sensitivity(:,j);
        count_sensitivity = count_sensitivity + 1;
    end
    if specificity(:,j) ~= Inf
       Specificity = Specificity + specificity(:,j);
       count_specificity = count_specificity + 1;
    end
    if f_measure(:,j) ~= Inf     
       F_measure = F_measure + f_measure(:,j);
       count_f_measure = count_f_measure + 1;
    end
    if false_alarm_rate(:,j) ~= Inf
       False_alarm_rate = False_alarm_rate + false_alarm_rate(:,j);
       count_false_alarm_rate = count_false_alarm_rate + 1;
    end
end

%Calculate the average of each evaluation variable
Accuracy = Accuracy / double(fold_size(1,1));
CV_Sequence = [CV_Sequence; 'Average'];
CV_Accuracy = [CV_Accuracy; Accuracy];
TP = int16(TP / double(fold_size(1,1)));
FP = int16(FP / double(fold_size(1,1)));
TN = int16(TN / double(fold_size(1,1)));
FN = int16(FN / double(fold_size(1,1)));
Recall = Recall / count_recall;
Precision = Precision / count_precision;
Sensitivity = Sensitivity / count_sensitivity;
Specificity = Specificity / count_specificity;
F_measure = F_measure / count_f_measure;
False_alarm_rate = False_alarm_rate / count_false_alarm_rate;

%Build a table recording accuracy in every Features sets(with different number
%of f)
CV_AccuracyTable = table(CV_Sequence, CV_Accuracy);

%Show Accuracy table  in a figure
Accuracy_show = uifigure;
uitable(Accuracy_show, 'Data', CV_AccuracyTable);

%Build a confusion matrix
Confusion_Matrix = {'Actual_Face'; 'Actual_NonFace'};
Predict_Face = {TP; FP};
Predict_NonFace =  {FN; TN};
Confusion_Table = table(Confusion_Matrix, Predict_Face, Predict_NonFace);

%Show confusion matrix in a figure
Confusion_matrix_show = uifigure;
uitable(Confusion_matrix_show, 'Data', Confusion_Table);

%Record test images size and precision, recall, specificity, etc in a matrix
Evaluation_Name = {'Accuracy';'Test images size'; 'Recall'; 'Precision'; 'Specificity';'Sensitivity'; 'F-measure'; 'False alarm rate'};
testLabs_fold_size = size(testLabs_fold);
Values = {Accuracy; testLabs_fold_size(1,1); Recall; Precision; Specificity; Sensitivity; F_measure; False_alarm_rate};
Precision_Sensitivity = table(Evaluation_Name, Values);

%Show test images size and precision, recall, specificity, etc in a figure
Precision_Sensitivity_show = uifigure;
uitable(Precision_Sensitivity_show, 'Data', Precision_Sensitivity);

<<<<<<< Updated upstream
%Training Data with all Features
Features = [trainFeatures_fold; testFeatures_fold];
Labels = [trainLabs_fold; testLabs_fold];
modelNN = NNtraining(Features, Labels);

%This is the end of model building, testing and evaluation for
%Cross-validation

end
=======

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
% %Groundtruth
% %im1 = 7 faces
% %im2 = 15 faces
% %im3 = 8 faces
% %im4 = 57?
% solutionTruth = [];
%              
% comparison = (predictedFullImage==solutionTruth);
% Accuracy = sum(sum(comparison))/ (size(comparison,1)*size(comparison,2))
% >>>>>>> Stashed changes
