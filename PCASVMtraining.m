function model = PCASVMtraining(images, labels, meanX, eigenvectors)



    %binary classification
    model.type='binary';
    
    %SVM software requires labels -1 or 1 for the binary problem
    labels(labels==0)=-1;

    %% kernel = gaussian
    kerneloption= 100; 
    kernel='gaussian';
    lambda = 1e-4;    %Don't go more than 0.01
    C = 1;

    %% kernel = polynomial
%     kernel='poly';
%     kerneloption=3;
%     lambda =10;  
%     C = 1;

%% Calculate the support vectors
    
    [xsup,w,w0,pos,tps,alpha] = svmclass(images,labels,C,lambda,kernel,kerneloption); 

    % create a structure encapsulating all teh variables composing the model
    model.xsup = xsup;
    model.w = w;
    model.w0 = w0;

  
    model.param.kerneloption=kerneloption;
    model.param.kernel=kernel;
    
    %Need to carry these along for the testing stage in the detector
    model.meanX = meanX;
    model.eigenvectors = eigenvectors;