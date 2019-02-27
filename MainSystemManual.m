clear all
close all

%% For loadFaceImages:

%If we use, we should only have the sampling at 1 so we can use as much training data as
%possible. Only alter this if we are doing a processing method which takes
%a long time. This is the easy way to extract feature vectors from the image but we'll use the
%manual version for now because it may give us more flexibility, with pre-processing images (HistEq) 
%and sampling (for example if we want to train with 80% of the images and test
%with 20%. With the loadFaceImages function the testing and training samples images are predetermined.

%TL:DR This is the automatic method of feature vector extraction. See MainSystemAutomatic for a better explanation
%[features, labs] = loadFaceImages('face_train.cdataset', 1); 

%% Below is a manual way to do feature extraction.
%It's could be more useful as it allows us to pre-process the images and choose the sampling method manually,
%rather than rely on the arbitrary sampling in the loadFaceImages function

%% Full image feature descriptor - Manual Feature Extraction
faceImages = zeros(69,27,18); % initialise 2D array to hold 69 27x18 images 
path = './images/face/*.png';
files = dir(path);
count = 0;
for file = files'
    count = count + 1;
    faceIm = imread(strcat('./images/face/', file.name));
    
    %faceIm = enhanceContrastHE(uint8(faceIm));
    %Hist Equalisation. For the moment it doesn't give us improved results
    
    %faceIm = enhanceContrastALS(uint8(faceIm));
    %Automatic Linear Stretching. For the moment it doesn't work. Throws
    %matirx dimension error
    
    faceImages(count, :, :) = faceIm;
end
%faceFeatures is now an array of face images
faceFeatures = reshape(faceImages, 69, 486);
%The above will create an array of 69 images with 486 pixels as a feature
%vector

nonFaceImages = zeros(59,27,18); % initialise 2D array to hold 59 27x18 images 
path1 = './images/non-face/*.png';
files1 = dir(path1);
count = 0;
for file = files1'
    count = count + 1;
    nonFaceIm = imread(strcat('./images/non-face/', file.name));
    %nonFaceIm = enhanceContrastHE(uint8(nonFaceIm));
    %Hist Equalisation. For the moment it doesn't give us improved results
    %nonFaceIm = enhanceContrastALS(uint8(nonFaceIm));
    %Automatic Linear Stretching. For the moment it doesn't work. Throws
    %matirx dimension error
    nonFaceImages(count,:, :) = nonFaceIm;
end
%nonFaceFeatures is now an array of face images. The array order will not
%necessarily correspond to the numbering of the files in the directory
nonFaceFeatures = reshape(nonFaceImages, 59, 486);

%Now we combine the feature vectors for the 69 raw face images and the 59 raw non-face
%images into one overall structure. 
totalFeatureV = [faceFeatures(:,:); nonFaceFeatures(:,:)]; 


%Label is 1 for face images and 0 for non-face images.
totalLabels(1:69) = 1;
totalLabels(70:128) = 0;

%Now we create a dataset which contains the totality of all the images. The
%first column will be each image's respective labels and the rest will be
%the feature vector which we'll do the machine learning on. We're doing
%this so when we split the data into training and testing samples we can
%keep consistent the labels corresponding to the appropriate feature
%vector. This may be useful when we attempt different, more complex
%sampling methods later (e.g. half/half, cross validation etc.)

totalData = [totalLabels.' totalFeatureV]; %128 images in total

%% Now we split this overall data into training and testing sets. 
%For the moment I'll have approx 80% for training and 20% for testing (we can change this later).
%So we'll put the first 80% face and non-face images in training (first 55 for face and 
%47 for non-face) and put the rest in testing

trainingSet = [totalData(1:55,:) ; totalData(70:117,:)]; %103 training images
testingSet = [totalData(56:69,:) ; totalData(118:128,:)]; %25 testing images

trainingLabels = trainingSet(:,1);
trainingFeatures = trainingSet(:,2:487);

testingLabels = testingSet(:,1);
testingFeatures = testingSet(:,2:487);

%% For visualization purposes, we display the first 10 face images
figure,
colormap(gray),
sgtitle("Displaying the first 10 Face images. Not Pre-Processed"),
for i=1:10

    
    Im = reshape(totalData(i,2:487),27,18);
    subplot(2,5,i), imagesc(Im), title(['label: ',labelToDescription(totalData(i,1))])
    axis off
    
end
%% We also display the first 10 non-face images
figure,
colormap(gray),
sgtitle("Displaying the first 10 non-face images. Not Pre-Processed"),
for i=70:79 
    Im = reshape(totalData(i,2:487),27,18);
    subplot(2,5,i-69), imagesc(Im), title(['label: ',labelToDescription(totalData(i,1))]) 
    axis off
end

%% Pre-Processing the images with Histogram Equalisation
% %We can use Histogram Equalisation to enhance the quality of our training
% %images. We don't require segmentation so histogram equalisation shouldn't
% %be a problem. Histogram Equalisation is also beneficial from the perspective
% %that parameters don't need to be tuned
% 
% % Noise filtering is a bad idea, whether it's median filter or a low pass
% % filter. Too much loss of detail plus the images have too small dimensions
% % for noise filtering. Each pixel is too important to get wrong
% 
% figure,
% colormap(gray),
% sgtitle("Comparing the first 5 face images. Not Pre-Processed vs Pre-Processed"),
% for i=1:5
% 
%     
%     Im = reshape(totalData(i,2:487),27,18);
%     OpIm = enhanceContrastHE(uint8(Im));
%     subplot(2,5,i), imagesc(Im), title(['label: ',labelToDescription(totalData(i,1))])
%     subplot(2,5,i+5), imagesc(OpIm), title('Pre-Processed')
%     axis off
%     
% end
% 
% figure,
% colormap(gray),
% sgtitle("Comparing the first 5 face images. Not Pre-Processed vs Pre-Processed"),
% for i=70:74 
%     Im = reshape(totalData(i,2:487),27,18);
%     OpIm = enhanceContrastHE(uint8(Im));
%     subplot(2,5,i-69), imagesc(Im), title(['label: ',labelToDescription(totalData(i,1))])
%     subplot(2,5,i-64), imagesc(OpIm), title('Pre-Processed')
%     axis off
% end


%% Note: 
%Comment out either NN or SVM related section depending on what method
%you're using on the script run. These sections are clearly marked

%% SVM Visualisation

[U,S,X_reduce] = pca(totalFeatureV,3);
imean=mean(totalFeatureV,1);
X_reduce=(totalFeatureV-ones(size(totalFeatureV,1),1)*imean)*U(:,1:3);

figure, hold on
colours= ['r.'; 'g.'; 'b.'];
count=0;
for i=min(totalLabels):max(totalLabels)
    count = count+1;
    indexes = find (totalLabels == i);
    plot3(X_reduce(indexes,1),X_reduce(indexes,2),X_reduce(indexes,3),colours(count,:))
end

%% SVM Training

%We have the provided SVMtraining model at our disposal. Can use this version
%or the one in the practicals. There's no real difference just the one in
%the practical has slightly differently tuned parameters and also controls
%for the case that a person's matlab version doesn't have a pre installed
%SVM toolkit

modelSVM = SVMtraining(trainingFeatures, trainingLabels); 

%After calculating the support vectors with our training method, we can draw them in the previous
%visualisation

hold on
%transformation to the full image to the best 3 dimensions 
imean=mean(trainingFeatures,1);
xsup_pca=(modelSVM.xsup-ones(size(modelSVM.xsup,1),1)*imean)*U(:,1:3);
% plot support vectors
h=plot3(xsup_pca(:,1),xsup_pca(:,2),xsup_pca(:,3),'go');
set(h,'lineWidth',5)


%% NN visualisation

% %Below produces a figure which demonstrates with PCA how non-face images
% %and face images are split in the feature space. Although the small amount
% %of features here means that there is a poor separation displayed in
% %the feature space, it's still useful for us to get a rough idea as to 
% %how separated these two categories are in the feature space
% 
% [U,S,X_reduce] = pca(totalFeatureV,3);
% 
% %This is just to help us visualise the difference in the faces images and non-face images. 
% figure, title("PCA visualising difference between face images and non-face images in the feature space"), hold on
% colours= ['r.'; 'g.'];
% count=0;
% for i=min(totalLabels):max(totalLabels)
%     count = count+1;
%     indexes = find (totalLabels == i);
%     plot3(X_reduce(indexes,1),X_reduce(indexes,2),X_reduce(indexes,3),colours(count,:))
% end
% 
% %% NN Training 
% modelNN = NNtraining(trainingFeatures, trainingLabels);

% Testing
for i=1:size(testingFeatures,1)
    
    testnumber= testingFeatures(i,:);
    
    %% NN model
    %classificationResult(i,1) = NNTesting(testnumber, modelNN);
    %Accuracy is 0.7600 for NN. With no resampling methods, just manual
    %feature extraction. Weird that it's this way because when it's tested
    %with KNN and K is set to 1 we get an accuracy of 0.6
    
    %% KNN model
    %classificationResult(i,1) = KNNTesting(testnumber, modelNN, 5);
    %Accuracy is between 0.56 and 6 for different values of K
    
    %% SVM Model
    classificationResult(i,1) = SVMTesting(testnumber,modelSVM);
    %Accuracy with no changed parameters is 0.8667

end

%% This bit is just to record a table for all of the accuracies for varying amounts of K in KNN
%Could be useful for the report later

% accuracyForEachK(20,2) = 0;
% accuracyForEachK(:,1) = 1:20;
% for k = 1:20
%    
%     for i = 1:size(testingFeatures,1)
%         testnumber= testingFeatures(i,:);
%         classificationResult(i,1) = KNNTesting(testnumber, modelNN, k);
%     end 
%     
%     comparison = (testingLabels==classificationResult);
%     Accuracy = sum(comparison)/length(comparison);
%     accuracyForEachK(k,2) = Accuracy; 
%     %accuracyForEachK is the final table holding the accuracy of the model
%     %for each value of K from 1:20
% end
% 
% accuracyForEachK

%% Evaluation

% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the testing image
comparison = (testingLabels==classificationResult);

%Accuracy is the most common metric. It is defiend as the number of
%correctly classified samples/ the total number of tested samples
Accuracy = sum(comparison)/length(comparison)


%% We display at most 25 of the correctly classified images
figure, 
sgtitle('Correct Classification'),
colormap(gray)
count=0;
i=1;
while (count<25)&&(i<=length(comparison))
   
    if comparison(i)
        count=count+1;
        subplot(5,5,count)
        Im = reshape(testingFeatures(i,:),27,18);
        imagesc(Im)
        axis off
    end
    
    i=i+1;
    
end


%% We display at most 25 of the incorrectly classified images
figure
sgtitle('Wrong Classification'),
colormap(gray)
count=0;
i=1;
while (count<25)&&(i<=length(comparison))
    
    if ~comparison(i)
        count=count+1;
        subplot(5,5,count)
        Im = reshape(testingFeatures(i,:),27,18);
        imagesc(Im)
        title(labelToDescription(classificationResult(i)))
        axis off
    end
    
    i=i+1;
    
end
