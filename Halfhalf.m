function [trainFeatures, testFeatures] = Halfhalf(trainFeatures, testFeatures, trainLabs, testLabs)



%Combine two set of Features together
Features = trainFeatures + testFeatures;

%Number of total features
n = numel(trainLabs + testLabs);

%Divided Features into train and test set in average
%Consider the situation that n is an odd number
if mod(n,2)==0
        trainFeatures = select(Features, 1:n/2);
        testFeatures = select(Features, n/2+1:n);
elseif mod(n,2)==1
        trainFeatures = select(Features, 1:(n+1)/2);
        testFeatures = select(Features, (n+1)/2+1:n);
        
        
        



end