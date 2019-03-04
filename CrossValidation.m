function [TrainFeatures, TestFeatures, trainLabs, testLabs] = CrossValidation(TrainFeatures, TestFeatures, trainLabs, testLabs, n)

%Total label length of trainFeatures and testFeatures
length = size(trainLabs) + size(testLabs);

%Total Features of TrainFeatures and TestFeatures
Features = [TrainFeatures; TestFeatures];

%Total labels of TrainFeatures and TestFeatures
Labs = [trainLabs;testLabs];

%Reset TrainFeatures and TestFeatures to be 0
%Reset TrainLabs and TestLabs to be 0
%Later these will be used in build the new order of features and labels
TrainFeatures = [];
TestFeatures = [];
trainLabs = [];
testLabs = [];


%Number of features in each fold
%Consider the situation that remainder exists
if(mod(length(1,1),n) ~= 0)
    fold = int16(length(1,1) / (n - 1) - mod(length(1,1) / (n - 1), 1)) + 1;
else
    fold = int16(length(1,1) / n);
end


%A pointer that record the number of Features have been filled in
indice = 1;


for i = 1:n
    
    %Break from the loop if all Features have be filled in
    if indice > length
        break
    end
    %Decide test Feature location in each fold using random number
    test_indice = randi([1 fold(1,1)],1);
    
    %Features in each fold
    for j = (i - 1) * fold(1,1) + 1 : (i) * fold(1,1)
        
        %Break from the loop if all Features have be filled in
        if indice > length
            break
        end
        
        %Check if it comes to test feature's location and decide whether to
        %place the current feature, trainFeatures, or testFeatures
        if j ==(i - 1) * fold(1,1) + test_indice
            TestFeatures = [TestFeatures;Features(indice,:)];
            testLabs = [testLabs;Labs(indice,:)];
            indice  = indice + 1;
            continue
        end
        
        TrainFeatures = [TrainFeatures;Features(indice,:)];
        trainLabs = [trainLabs;Labs(indice,:)];
        indice = indice + 1;
    end
end

end
        