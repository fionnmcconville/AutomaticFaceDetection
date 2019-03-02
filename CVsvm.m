%function predic = CVcsv(pred)
%To perform cross-validation with svm learning way, there is no need to
%seperate dataset into train and test
%[cvFeatures1, cvLabs1] = loadFaceImages('face_test.cdataset', 1);
%[cvFeatures2, cvLabs2] = loadFaceImages('face_train.cdataset' ,1);
%cvFeatures =  cvFeatures1 + cvFeatures2;

% first we check if the problem is binary classification or multiclass
%if max(labels)<2
    %binary classification
    %model.type='binary';
    
    %SVM software requires labels -1 or 1 for the binary problem
    %labels(labels==0)=-1;


%We create indices for 10-fold cross-calidation
%indice = crossvalind('Kfold', labels, 10);

%Initialize an object to measure the performance of the classifier.
%cp = classperf(labels);

%for i = 1:10
    %test = (indice == i);
    %train = ~test;
    %class = classify(meas(test,:),meas(train,:),labels(train,:));
    %classperf(cp,class,test);
%end
%pred = cp.ErrorRate

%Code above is another try, delete it before finalisation

function model = CVsvm(images, labels)

%Build svm model with  images
svm = fitcsvm(images,labels);

%Build a cross validation model with svm
model = crossval(svm);

end