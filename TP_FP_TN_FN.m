function [TP, FP, TN, FN] = TP_FP_TN_FN(testLabs, classificationResult)


%Initialisation 
%True Positive Class, 
%False Positive Class, 
%True Negative Class,
%False Negative Class
TP = 0;
FP = 0;
TN = 0;
FN = 0;

%Record position of elements that predicted predicted correctly
True = testLabs == classificationResult;

for i = 1:length(testLabs)
    if True(i) == 1 && testLabs(i) == 1
        
        %Predicted correctly and predicted as Face image
        TP = TP + 1;
        
    elseif True(i) ~= 1 && testLabs(i) == 1
        
        %Predicted wrongly and predicted as Face image
        FP = FP + 1;
        
    elseif True(i) == 1 && testLabs(i) ~= 1
        
        %Predicted correctly and predicted as Non-Face image
        TN = TN + 1;
        
    else
        
         %Predicted wrongly and predicted as Non-Face image
        FN = FN + 1;
        
    end
end

end