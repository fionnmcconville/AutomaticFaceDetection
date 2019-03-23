function [detections] = SlidingWindow(I)
%SLIDINGWINDOW provides the detections using the model and a sliding window

%   This function will apply a sliding window to the input image and scan
%   for detections of the objects that the model has been trained to
%   detect.

%We load the classification model of our choice
modelLoad = load('detectorModelSVM.mat');
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

for x = 1:8
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
                digitIm = enhanceContrastALS(uint8(digitIm));
                digitIm = gabor_feature_vector(uint8(digitIm));

                %display the individually segmented digits
%                 subplot(samplingX,samplingY,digitCounter)
%                 imshow(digitIm)

                %we reshape the digit into a vector
                digitIm = reshape(digitIm, 1, []);

                [prediction, maxi] =  SVMTesting(digitIm,model);
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
%     figure
%     imagesc(map);
end
detections = object_detections;
end

