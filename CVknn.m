function  model = CVknn(images, labels)

%Build knn model with train images
%Declare number of neighbors as 5
knn = fitcknn(images, labels, 'NumNeighbours', 5);

%Build a cross validation model with knn
model = crossval(knn);