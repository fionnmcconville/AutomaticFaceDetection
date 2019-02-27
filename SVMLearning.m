clear all
close all

[features, labs] = loadFaceImages('face_train.cdataset', 10); 

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
%faceFeatures is now an array of face images
faceFeatures = reshape(faceImages, 69, 486);
%The above will create an array of 69 images with 486 pixels as all of
%it's pixel values. Been checked and it is correct.
% i.e flattenedArray(i,j) = faceImages(i, j)


nonFaceImages = zeros(59,27,18); % initialise 2D array to hold 59 27x18 images 
path1 = './images/non-face/*.png';
files1 = dir(path1);
count = 0;
for file = files1'
    count = count + 1;
    faceIm = imread(strcat('./images/non-face/', file.name));
    nonFaceImages(count,:, :) = faceIm;
end
%nonFaceFeatures is now an array of face images. The array order will not
%necessarily correspond to the numbering of the files in the directory
nonFaceFeatures = reshape(nonFaceImages, 59, 486);

%Now we combine the feature vectors for the cdataset 100 face images, the
%69 raw face images and the 59 raw non-face images into one overall structure. 
totalFeatureV = [features(:,:); faceFeatures(:,:); nonFaceFeatures(:,:)];

%label 1 for face images and 0 for non-face images.
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
    axis off
    
end
 %We also display the first 10 non-face images
figure,
colormap(gray),
sgtitle("Displaying the first 10 non-face images"),
for i=1:10 
    Im = reshape(nonFaceFeatures(i,:),27,18);
    subplot(2,5,i), imagesc(Im), title(['label: ',labelToDescription(totalLabels(169 + i))])  
    axis off
end

%We can use the above apparatus to check the effects of possible
%pre-processing later

[U,S,X_reduce] = pca(totalFeatureV,3);
imean=mean(totalFeatureV,1);
X_reduce=(totalFeatureV-ones(size(totalFeatureV,1),1)*imean)*U(:,1:3);

figure, hold on
colours= ['r.'; 'g.'; 'b.'; 'k.'; 'y.'; 'c.'; 'm.'; 'r+'; 'g+'; 'b+'; 'k+'; 'y+'; 'c+'; 'm+'];
count=0;
for i=min(totalLabels):max(totalLabels)
    count = count+1;
    indexes = find (totalLabels == i);
    plot3(X_reduce(indexes,1),X_reduce(indexes,2),X_reduce(indexes,3),colours(count,:))
end

%We have the provided SVMtraining model at our disposal. Can use this version
%or the one in the practicals
modelSVM = SVMtraining(totalFeatureV, totalLabels.'); 
%Transposed labels list. The model throws up an error if the amount of rows
%in ar1 and arg2 aren't equal

%After calculating the support vectors, we can draw them in the previous
%image

hold on
%transformation to the full image to the best 3 dimensions 
imean=mean(totalFeatureV,1);
xsup_pca=(modelSVM.xsup-ones(size(modelSVM.xsup,1),1)*imean)*U(:,1:3);
% plot support vectors
h=plot3(xsup_pca(:,1),xsup_pca(:,2),xsup_pca(:,3),'go');
set(h,'lineWidth',5)

%% Testing

[testfeatures, testlabs] = loadFaceImages('face_test.cdataset'); 

for i=1:size(testfeatures,1)
    
    testnumber= testfeatures(i,:);
    classificationResult(i,1) = SVMTesting(testnumber,modelSVM);
    %Accuracy with no changed paramters is 0.8667

end

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
