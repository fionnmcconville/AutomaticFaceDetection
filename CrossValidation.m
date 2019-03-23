 function [TrainFeatures, TestFeatures, trainLabs, testLabs] = CrossValidation(TrainFeatures, TestFeatures, trainLabs, testLabs, n, fold_size, length, f)

%%Cross-validation: For each classification: Divide Features into n folds, and select one of features
%%in each fold as test features, perform classification fold times with
%%each classification's test features location different from each other(1
%%test features in the ending may repeat to suit the size of new features)
%%.And then count the average accuracy of each classification's model.


%Combine Features of TrainFeatures and TestFeatures
Features = [TrainFeatures; TestFeatures];

%Total labels of TrainFeatures and TestFeatures
Labs = [trainLabs;testLabs];

%Reset TrainFeatures and TestFeatures to be 0
%Reset TrainLabs and TestLabs to be 0
%Later these will be used in build the new order of features and labels
%Record n classifications
TrainFeatures = [];
TestFeatures = [];
trainLabs = [];
testLabs = [];


%A pointer that record the number of Features have been filled in
indice = 1;

for i = 1:n
    
    %Break from the loop if all Features have be filled in
    if indice == length(1,1)
        break
    end
    %Decide test Feature location in each fold with input 'f'
    test_indice = f;
    
    %Features in each fold
    for j = (i - 1) * fold_size + 1 : i * fold_size 
        
        %Break from the loop if all Features have be filled in
        if indice == length(1,1)
            break
        end
        
        %Check if it comes to test feature's location and decide whether to
        %place the current feature, trainFeatures, or testFeatures
        if j == (i - 1) * fold_size + test_indice
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

   