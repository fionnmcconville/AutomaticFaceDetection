function [images, labels] = augmentImages(features, oldlabels)
%List of full image feature vectors with corresponding labels are
%passed into function and 10 different versions of the image
%are created in code below. These 10 images are then converted to 1D feature vectors
%and stored in a list which is the output
images=[];
labels = [];
for i=1:size(features,1)
    
    label = oldlabels(i);   %Getting the label for the image
    Im = features(i,:);  %extracting the single feature vector from a list of feature vectors
    I = reshape(Im,27,18); %Converting feature vector in an image - needed for augmentation below
    
    % if label==1
    vector = reshape(I,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    images= [images; vector];
    labels= [labels; label];

    Itemp =fliplr(I);
    vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    images= [images; vector];
    labels= [labels; label];


    Itemp = circshift(I,1);
    vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    images= [images; vector];
    labels= [labels; label];


    Itemp = circshift(I,-1);
    vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    images= [images; vector];
    labels= [labels; label];


    Itemp = circshift(I,[0 1]);
    vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    images= [images; vector];
    labels= [labels; label];


    Itemp =circshift(I,[0 -1]);
    vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    images= [images; vector];
    labels= [labels; label];


    Itemp = circshift(fliplr(I),1);
    vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    images= [images; vector];
    labels= [labels; label];


    Itemp =circshift(fliplr(I),-1);
    vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    images= [images; vector];
    labels= [labels; label];


    Itemp = circshift(fliplr(I),[0 1]);
    vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    images= [images; vector];
    labels= [labels; label];


    Itemp = circshift(fliplr(I),[0 -1]);
    vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    images= [images; vector];
    labels= [labels; label];
end

end


