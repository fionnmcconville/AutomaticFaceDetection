function [trainFeatures, testFeatures, trainLabs, testLabs] = Halfhalf(trainFeatures, testFeatures, trainLabs, testLabs, percentage)



%Combine two set of Features and labels together
Features = [trainFeatures ; testFeatures];
Labs = [trainLabs; testLabs];

%Number of total features
n = size(trainLabs) + size(testLabs);

%Reset TrainFeatures and TestFeatures to be 0
%Reset TrainLabs and TestLabs to be 0
%Later these will be used in build the new order of features and labels
trainFeatures = [];
testFeatures = [];
trainLabs = [];
testLabs = [];


%Get new test Features size based on percentage input
num_test = int16(percentage*n(1,1));

%Create a set to store indices of new test Features in total Features set
test_loc_set = randperm(n(1,1), num_test);

%Sort elements in test_index to be ascending 
test_loc_set = sort(test_loc_set);

%Indix point to current location of test Features in total Features set using random number
test_index = 1;

%Divided Features into train and test set in average and in random
for i = 1:n(1,1)
    if(test_loc_set(:,test_index)==i)
        testFeatures = [testFeatures;Features(i,:)];
        testLabs = [testLabs;Labs(i,:)];
        if(test_index < num_test)
        test_index = test_index + 1;
        end
    else
        trainFeatures = [trainFeatures;Features(i,:)];
        trainLabs = [trainLabs;Labs(i,:)];
    end
end

end