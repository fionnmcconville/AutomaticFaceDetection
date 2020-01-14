function model = SVMtraining(images, labels)


% first we check if the problem is binary classification or multiclass
if max(labels)<2
    %binary classification
    model.type='binary';
    
    %SVM software requires labels -1 or 1 for the binary problem
    labels(labels==0)=-1;

     %Initilaise and setup SVM parameters
    %C is the margin of error of the support vectors. Basically a smaller C
    %will have a wider distance of boundary to support vector so at too
    %small a value it may encapsulate too many errors on the data. However a
    %very large C may tilt the boundary line too much and not capture the
    %trend of the graph as well. We can use cross-validation maybe to pick
    %a good value for C. A smaller C gives us a better separating boundary
    %as the line separates the points better. However the margin of error
    %has too many points in it. We want a point-free margin for the best
    %model. Inf makes the classifier as strict as possible which may not
    %work for some kernel functions
        
    %% kernel = polynomial - gabor
    %Works really well with gabor features. Relatively Quick method.
    kernel='poly';
    kerneloption=4;
    lambda = 1e-20;  %Keep as 1e-20 for gabor. Lots of features need small lambda. Gives better results on Detector
    C = 8e8; % higher C = more time. Stricter classification
%     %Changing kerneloption has effect on acc. kerneloption is the degree of
    %the polynomial I think (Should find out what EXACTLY this means. Is optimum when = 2 (Accuracy is 0.925)
    
    %% kernel = gaussian - gabor
    %kerneloption is kernel scale. Should be quite large? If accuracy is
    %0.667 then it's not large enough
    
%     kernel='gaussian';
%     kerneloption= 4767;
%     lambda = 1e-20;         %A lot of features = small lambda. Lambda is the weight of quadratic function.
%     C = 1200;


    % Calculate the support vectors
    [xsup,w,w0,pos,tps,alpha] = svmclass(images,labels,C,lambda,kernel,kerneloption); 

    % create a structure encapsulating all teh variables composing the model
    model.xsup = xsup;
    model.w = w;
    model.w0 = w0;

    model.param.kerneloption=kerneloption;
    model.param.kernel=kernel;
    
    
else
    %multiple class classification
     model.type='multiclass';
    
    %SVM software requires labels from 1 to N for the multi-class problem
    labels = labels+1;
    nbclass=max(labels);
    
    %Initilaise and setup SVM parameters
    lambda = 1e-20;  
    C = 100000;
    kerneloption= 5;
    kernel='gaussian';
    
    % Calculate the support vectors
    [xsup,w,b,nbsv]=svmmulticlassoneagainstall(images,labels,nbclass,C,lambda,kernel,kerneloption,1);
    
    % create a structure encapsulating all teh variables composing the model
    model.xsup = xsup;
    model.w = w;
    model.b = b;
    model.nbsv = nbsv;

    model.param.kerneloption=kerneloption;
    model.param.kernel=kernel;
    
end

end

