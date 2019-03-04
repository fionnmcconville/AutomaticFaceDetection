function [trainFeatures, testFeatures, trainLabs, testLabs] = Halfhalf(trainFeatures, testFeatures, trainLabs, testLabs)



%Combine two set of Features and labels together
Features = [trainFeatures ; testFeatures];
Labs = [trainLabs; testLabs];


%Number of total features
n = size(trainLabs) + size(testLabs);

%Divided Features into train and test set in average
%Consider the situation that n is an odd number
if mod(n(1,1),2)==0
        trainFeatures = Features(1:n/2,:);
        testFeatures = Features(n/2+1:n,:);
        trainLabs = Features(1:n/2,:);
        testLabs = Features(n/2+1:n,:);
elseif mod(n(1,1),2)==1
        trainFeatures = Features(1:(n+1)/2,:);
        testFeatures = Features((n+1)/2+1:n,:);
        trainLabs = Features(1:n/2,:);
        testFeatures = Features((n+1)/2+1:n,:);
end
        
        



end