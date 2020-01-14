function [detections] = SlidingWindow(I)
%SLIDINGWINDOW provides the detections using the model and a sliding window

%   This function will apply a sliding window to the input image and scan
%   for detections of the objects that the model has been trained to
%   detect.

addpath .\SVM-KM\

%We load the classification model of our choice
%modelLoad = load('detectorModelSVM.mat');
modelLoad = load('PolynomialModelSVM.mat');
model = modelLoad.modelSVM;

% convert image to gray scale if not already
[rows, columns, numberOfColorChannels] = size(I);
if numberOfColorChannels > 1
  I=rgb2gray(I);
end

%sliding window values on a *1 scale
windowWidth = 20;
windowHeight = 25;

%provide different scales to the image and loop through each scale
multiplier = 0.25;
stepSize = 10;

%create structure to hold all detections
object_detections = [];

for x = 4:8
    Im = imresize(I, multiplier*x);
    digitCounter=0;
    winWidth = floor(windowWidth*(multiplier*(x+multiplier)));
    winHeight = floor(windowHeight*(multiplier*(x+multiplier)));

    %for each digit within the image, 
    county=0;
    for r=1:stepSize:size(Im,1)
        county=county+1;
        countx=0;
        for c= 1:stepSize:size(Im,2)
            countx=countx+1;
            %ensures that 
            if (c+winWidth-1 <= size(Im,2)) && (r+winHeight-1 <= size(Im,1))
                digitCounter = digitCounter+1;

                %crop the digit
                digitIm = Im(r:r+winHeight-1, c:c+winWidth-1);

                %resample them into a 27x18 imaGE
                digitIm = imresize(digitIm, [27 18]);
                %colormap(gray), imagesc(digitIm), drawnow;
                
                %Draw sliding window on image
                xCo = (r-1) / (multiplier*x);
                yCo = (c-1) / (multiplier*x);
                ht = xCo + winHeight / (multiplier*x);
                wd = yCo + winWidth / (multiplier*x);  
                p = plot([yCo yCo wd wd yCo],[xCo ht ht xCo xCo], 'r'); 
                drawnow;
                delete(p);
                
                %% Gabor features - Comment out if you're using PCA
                digitIm = enhanceContrastALS(uint8(digitIm));
                digitIm = gabor_feature_vector(uint8(digitIm)); %Gabor Features
                digitIm = digitIm(:)'; %Turn back into feature vector
                [prediction, maxi] =  SVMTesting(digitIm,model);

                %% PCA - Comment out if using Gabor
%                 digitIm = digitIm(:)';
%                 Xpca = (double(digitIm) - model.meanX) * model.eigenvectors; %PCA Features
%                 [prediction, maxi] =  SVMTesting(Xpca,model);
%                 
                %% Detections
                if prediction > 0.5
                    xCord = (r-1) / (multiplier*x);
                    yCord = (c-1) / (multiplier*x);
                    height = winHeight / (multiplier*x);
                    width = winWidth / (multiplier*x);  
                    detected = [xCord, yCord, width, height, maxi];
                    object_detections = [object_detections; detected];
                end                
                map(county,countx) = maxi;
            end
        end        
    end
   
%      colormap(gray), imagesc(map), drawnow;
end
detections = object_detections;
end

