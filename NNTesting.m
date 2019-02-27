function prediction = NNTesting(testImage,modelNN)
%Final NN classification algorithm which calculates the eucledian distance
%(using a previously created function) between a test sample and each
%training sample already existing in modelNN.

closestDist = 1000000; %Initialise to a large value so that it will be eventually overwritten by a smaller value in the for loop below
predictionInd = 0;
for i = 1: size(modelNN.neighbours, 1)
    %Just going through all elements (1205) of
    %the sampling of training items in our dataset with th first dimension
    %of modelNN.neighbours
    trainImage = modelNN.neighbours(i,:);
    dist = EucledianDistance(trainImage, testImage);
    if dist > closestDist
        continue
    else
        closestDist = dist;
        predictionInd = i;
    end
end
prediction = modelNN.labels(predictionInd);
end


