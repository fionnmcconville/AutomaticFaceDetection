clear all
close all

%This creates an array of 100 feature vectors for the training dataset. All
%images represented in this dataset are faces. We can alter the sampling to
%have more data for our model, by either reducing the sampling or having
%none at all....
[features, labs] = loadFaceImages('face_train.cdataset', 10); 

%For some reason it has an output of all the pixel values for each image in the console
%A bit annoying but not too worrisome

%We don't just want to use the provided cdataset as it only has examples of
%face images. We will also use the raw images given to us - which has examples of faces and
%non faces - and convert the full image into a feature vector. A supervised
%learning model performs far better if it is given positive and negative
%examples

%Will most likely attempt to preprocess the images in some small way and
%see if it improves the accuracy of our models. Can preprocess the raw
%images and if there is an improvement then we can convert the provided
%cdataset face images to an image - preprocess them and then convert back
%to finally train our model

%% Full image feature descriptor
faceImages = zeros(69,27,18); % initialise 2D array to hold 69 27x18 images 
path = './images/face/*.png';
files = dir(path);
count = 0;
for file = files'
    count = count + 1;
    faceIm = imread(strcat('./images/face/', file.name));
    faceImages(count, :, :) = faceIm;
end
%faceImages is now an array of face images
faceFeatures = reshape(faceImages, 69, 486);
%The above would create an array of 69 images with 486 pixels as all of
%it's pixel values. Been checked and it is correct.
% i.e flattenedArray(i,j) = faceImages(i, j)


nonFaceImages = zeros(59,27,18); % initialise 2D array to hold 69 27x18 images 
path1 = './images/non-face/*.png';
files1 = dir(path1);
count = 0;
for file = files1'
    count = count + 1;
    faceIm = imread(strcat('./images/non-face/', file.name));
    nonFaceImages(count,:, :) = faceIm;
end
%nonFaceImages is now an array of face images. The array order will not
%necessarily correspond to the numbering of the files in the directory
nonFaceFeatures = reshape(nonFaceImages, 59, 486);

%Now we combine the feature vectors for the cdataset 100 face images, the
%69 raw face images and the 59 raw non-face images into one overall structure. 


totalFeatureV = [features(:,:); faceFeatures(:,:); nonFaceFeatures(:,:)];

%label 1 for face images and 0 for non-face images. (Could maybe think
%about encoding strings "face" and "non-face" as labels for clarity)?

totalLabels(1:69) = 1;
totalLabels(70:128) = 0;
totalLabels = [labs.', totalLabels]; %Transpose the labs vector so dimensions match

% For visualization purposes, we display the first 10 face images
figure,
colormap(gray),
sgtitle("Displaying the first 10 Face images"),
for i=1:10

    
    Im = reshape(faceFeatures(i,:),27,18);
    subplot(2,5,i), imagesc(Im), title(['label: ',labelToDescription(totalLabels(i))])
    
end
 %We also display the first 10 non-face images
figure,
colormap(gray),
sgtitle("Displaying the first 10 non-face images"),
for i=1:10 
    Im = reshape(nonFaceFeatures(i,:),27,18);
    subplot(2,5,i), imagesc(Im), title(['label: ',labelToDescription(totalLabels(169 + i))])  
end

%We can use the above apparatus to check the effects of possible
%pre-processing later


%Supervised training Nearest Neighbour function that takes the examples and infers a model
modelNN = NNtraining(totalFeatureV, totalLabels);


%% NN visualisation

%Below produces a figure which demonstrates with PCA how non-face images
%and face images are split in the feature space. Although the small amount
%of features here means that there is a poor separation displayed in
%the feature space, it's still useful for us to get a rough idea as to 
%how separated these two categories are in the feature space

[U,S,X_reduce] = pca(totalFeatureV,3);

%This is just to help us visualise the difference in the faces images and non-face images. 
figure, title("PCA visualising difference between face images and non-face images in the feature space"), hold on
colours= ['r.'; 'g.'];
count=0;
for i=min(totalLabels):max(totalLabels)
    count = count+1;
    indexes = find (totalLabels == i);
    plot3(X_reduce(indexes,1),X_reduce(indexes,2),X_reduce(indexes,3),colours(count,:))
end


%% testing
% Loading testing labels and testing examples of face images from face_test.cdataset
% It is very important that this images are different from the ones used in
% training or our results will not be reliable. The images we are testing
% with are all face images. For now we will use this as a test dataset, if
% perhaps later down the line we can get test items that are non-faces it
% would be good. Perhaps we can use less non-faces in the inital supervised
% training and use them instead here....

[testfeatures, testlabs] = loadFaceImages('face_test.cdataset'); 
%With a sampling of 5 we get 60 test items. Not sure how necessary this is
%yet, will have to know more about what happens when we get more testing
%samples fromthis cdataset. Is it just repeat values? Or new values each
%time? With no sampling we get 240 items

for i=1:size(testfeatures,1)
    
    testnumber= testfeatures(i,:); % We go through eachof the 211 test samples
    
    %% NN model
    %classificationResult(i,1) = NNTesting(testnumber, modelNN);
    %Accuracy is 0.9167 with a test sampling of 5
    %Accuracy is 0.7583 with no test sampling
    
    %% KNN model
    classificationResult(i,1) = KNNTesting(testnumber, modelNN, 15);
    %With k = 4 and k = 6. Accuracy with no test sampling is 0.8250
    %With k = 9 and k = 15. Accuracy is 0.8333
    
    
end

%Could perhaps do a for loop to test and display the accuracy of our model
%for each value of k up to 20 or 30. Then have the final model be the K
%with the optimum accuracy.

%% Evaluation

% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the testing image
comparison = (testlabs==classificationResult);

%Accuracy is the most common metric. It is defiend as the numebr of
%correctly classified samples/ the total number of tested samples
Accuracy = sum(comparison)/length(comparison)


%We display 25 of the correctly classified images
figure, 
sgtitle('Correct Classification'),
colormap(gray)
count=0;
i=1;
while (count<25)&&(i<=length(comparison))
   
    if comparison(i)
        count=count+1;
        subplot(5,5,count)
        Im = reshape(testfeatures(i,:),27,18);
        imagesc(Im)
    end
    
    i=i+1;
    
end


%We display 25 of the incorrectly classified images
figure
sgtitle('Wrong Classification'),
colormap(gray)
count=0;
i=1;
while (count<25)&&(i<=length(comparison))
    
    if ~comparison(i)
        count=count+1;
        subplot(5,5,count)
        Im = reshape(testfeatures(i,:),27,18);
        imagesc(Im)
        title(labelToDescription(classificationResult(i)))
    end
    
    i=i+1;
    
end


