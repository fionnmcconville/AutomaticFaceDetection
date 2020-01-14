function [Sort_Wrong_Record] = CV_Wrong_Frequency(Wrong_Record, trainingFeatures, testingFeatures)

%% Function for record the around 25 Features with highest Frequency to be falsely 
% recognise (in Cross-validation) and show it in a figure.

%Descend sort for Frequency of Features Sequences
Sort_Wrong_Record = int16.empty;

%Record number of Features' Wrong Classification totally
Size_Wrong_Record  = size(Wrong_Record);

if Size_Wrong_Record(1,1) ~= 0
Frequency_Wrong_Record = zeros(size(Wrong_Record));
for i = 1:Size_Wrong_Record(1,1)
Frequency_Wrong_Record(i) = sum(Wrong_Record==Wrong_Record(i));
end

Wrong_Record = [Wrong_Record, Frequency_Wrong_Record];

%Descend the frequency type
Frequency_Wrong_Record = sort(Wrong_Record(:,2), 'descend');

%Find out a series of Features with highest Wrongly classified frequency to be wrongly
%classsified
i = 1;
k = 0;

while k < 25 & i <= Size_Wrong_Record(1,1)
    %Detect if the Frequency is changed
    Current_value = Frequency_Wrong_Record(i,:);
    
    %Count how many wrongly classified Features has the same frequency
    count = 0;
    
     %An index array to record  which row including the aimed value
        index = find(Wrong_Record(:,2) == Current_value);
        
    
     while Current_value == Frequency_Wrong_Record(i,:)
         i = i + 1; 
         count = count + 1;   
        if i > Size_Wrong_Record(1,1)
            break
        end
         if Current_value ~= Frequency_Wrong_Record(i,:)
             break;
         end
        
     end

    %Record the Sequence by its frequency
    Sort = int16.empty;
    if count > 1
    for j = 1 : count       
      % Sort = [unique(index),Current_value * ones(index_size(1,1),1)];
        Sort = [Sort; Wrong_Record(index(j,1),1)];
    end
    else
        %Sort_Wrong_Record = [Sort_Wrong_Record; Wrong_Record(index(1,1),1)];
        Sort = [Sort; Wrong_Record(index(1,1),1)];
    end
        Sort = unique(Sort);
        %Size of Features with same frequency without repeatition
        Size_sort = size(Sort);
        
        k = k + Size_sort;
        
        if k <= 25
            a = zeros(Size_sort(1,1),1);
            a(:) = Current_value;
            Sort = [Sort, a];
            Sort_Wrong_Record = [Sort_Wrong_Record; Sort];
        else
            Sort = Sort(1:(Size_sort - k + 25),:);
            Size_sort = size(Sort);
            a = zeros(Size_sort(1,1),1);
            a(:) = Current_value;
            Sort = [Sort, a];
            Sort_Wrong_Record = [Sort_Wrong_Record; Sort];
        end
    end
          
Size_sort = size(Sort_Wrong_Record(:,1));

%Combine Features together
Features = [trainingFeatures;testingFeatures];

%Showing Features falsely(wrongly) classified most frequently in afigure
figure
title('Features misclassified most frequently'),
colormap(gray)
        for i = 1:Size_sort(1,1)
        subplot(5,5,i)
        a = Sort_Wrong_Record(i,1);
        Im = reshape(Features(a,:),27,18);
        imagesc(Im)
        title("Misclassified: "+Sort_Wrong_Record(i,2))
        axis off
       
        end
end
end
