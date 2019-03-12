function prediction = KNNTesting(testImage,modelNN, K)
%KNN classification algorithm which calculates the eucledian distance
%(using a previously created function) between a test sample and each
%training sample already existing in modelNN. Final K nearest neighbours
%will have a sorted set of eucledian distances and indices


Knearest(1:K) = 10000000 * rand(K,1); 
%Initialise each distance in Knearest to a large value so that it will be eventually
%overwritten by a smaller value in the for loop below. 

predictionInd(1:K) = 10;

for i = 1:size(modelNN.neighbours, 1)
%Just going through all elements of the training items in our dataset 
%with the first dimension of modelNN.neighbours
    trainImage = modelNN.neighbours(i,:);
    dist = EucledianDistance(trainImage, testImage);
    if dist > max(Knearest)
        continue
%The last element of Knearest is always the one we want to replace as it's the 
%largest value when the array is sorted. So element at K index of sortedKnearest will be
%replaced in Knearest. We may also just find the index of max Knearest
    else
        replaceIndex = find(Knearest == max(Knearest), 1, 'first'); %Make sure to have so as to avoid the case where we replace multiple of the same distance
        Knearest(replaceIndex) = dist; %Replace largest distance in array with new smaller eucledian distance    
        predictionInd(replaceIndex) = i;
    end
end

nearestNeighbours(1:K) = 10;
for i = 1:K
    nearestNeighbours(i) = modelNN.labels(predictionInd(i));
end
prediction = mode(nearestNeighbours);
end


