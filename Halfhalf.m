function [trainFeatures, testFeatures] = Halfhalf('face_train.cdataset', 'face_test.cdataset')


%Load all Features from dataset
[Features1, testLabs] = loadFaceImages('face_test.cdataset', 1);
[Features2, testLabs] = loadFaceImages('face_train.cdataset', 1);

%Combine two set of Features together
Features = Features1 + Features2;

%Divided Features into train and test set in average
trainFeatures = select(Features, 1:30);
testFeatures = select(Features, 31:59);

end