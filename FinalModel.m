clear all
close all

addpath .\SVM-KM\ 
%In order to use the provided SVM toolbox. We must add the path to the toolbox folder. 
%This folder is in prac 5


%% Utilizing the entire image dataset to train the model. Using all of the data for training should give a better classifier

%Loading in images
[trainFeatures, trainLabs] = loadFaceImages('face_train.cdataset', 1);
[testFeatures, testLabs] = loadFaceImages('face_test.cdataset', 1);

%Combining the datasets
features = [trainFeatures; testFeatures];
labs = [trainLabs; testLabs];

%Performing augmentation to increase the amount of images to train the model
[features, labs] = augmentImages(features, labs);

dsize = size(features, 1);
numFeatures = size(features,2);

%% Pre-Processing Images - Detector results are better with PCA if this is commented out
for i = 1:dsize
    Im = reshape(features(i,:),27,18);

    Im = enhanceContrastALS(uint8(Im));
    
    features(i,:) = Im(:)';  
end

%% Gabor Features - Comment out if using PCA
%Convert features to images so they can be processed by gabor_features function

% gaborFeatures = zeros(dsize, numFeatures * 40);
% for i = 1:dsize
%     Im = reshape(features(i,:),27,18);
%     
%     Im = gabor_feature_vector(uint8(Im)); %Produces 19,440 features 
%     gaborFeatures(i,:) = Im;
% end
% 
% modelSVM = SVMtraining(gaborFeatures, labs);
% save PolynomialModelSVM modelSVM
% %save GaussianModelSVM modelSVM
% 

 %% PCA Features - Comment out if using Gabor
 
%[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(features);
%modelSVM = PCASVMtraining(Xpca, labs, meanX, eigenVectors);
%save PCAGaussianModelSVM1 modelSVM
%save PCAPolynomialModelSVM modelSVM


 %% LDA Features - Comment out if using Gabor
 
[eigenVectors, eigenvalues, meanX, Xpca] = LDA(labs,[], features);
%modelSVM = PCASVMtraining(Xpca, labs, meanX, eigenVectors);
%save PCAGaussianModelSVM1 modelSVM
%save PCAPolynomialModelSVM modelSVM

%% Run to do a bayes optimisation search to determine best parameters for Gabor features
%bayesSearch(gaborFeatures, labs); 

%% Run to do a bayes optimisation search to determine best parameters for PCA features
bayesSearch(Xpca, labs); 
