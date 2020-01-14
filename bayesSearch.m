function model = bayesSearch(images, labels)

%SVM software requires labels -1 or 1 for the binary problem
labels(labels==0)=-1;


 %% Search for optimal params
 %Does a Bayes Optimisation search that finds parameters to minimise
 %the objective function for SVM. Uses KFold validation to test
params = hyperparameters('fitcsvm',images,labels);
params(1).Range = [1, 1e20]; % 'BoxConstraint'
params(2).Range = [1e-2, 1e5]; %'KernelScale'
params(3).Range = {'rbf' 'gaussian'}; %'KernelFunction'
params(3).Optimize = true;
% params(4).Range = [4,5]; %Polynomial Order
% params(4).Optimize = true;
opts = struct('Optimizer','bayesopt','Kfold', 5, 'SaveIntermediateResults', true, 'ShowPlots',true,'AcquisitionFunctionName','expected-improvement-plus');
modelFITSVM = fitcsvm(images, labels,'Solver', 'L1QP','OptimizeHyperparameters' ,params, 'HyperparameterOptimizationOptions',opts);
       
model.modelFITSVM = modelFITSVM;

    
