function [Recall, Precision, Specificity, Sensitivity, F_measure, False_alarm_rate]  = Precision_Sensitivity(TP, FP, TN, FN)

%These is a function to calculate value of Recall, Precision and stc. based
%on True Positive class, False Positive class,  False Negative class, True
%Negative class

%Recall
if (TP + FN) ~= 0
    Recall = TP / (TP + FN);
else
    Recall = 'Not exist';
end

%Precision
if (TP + FP) ~= 0
    Precision = TP / (TP + FP);
else 
    Precision = 'Not exist';
end

%Specificity
if (TN + FP) ~= 0
    Specificity = TN / (TN + FP);
else
    Specificity = 'Not exist';
end

%Sensitivity
Sensitivity  =Recall;

%F-measure
if(2 * TP + FN + FP) ~= 0
    F_measure = 2 * TP / (2 * TP + FN + FP);
else
    F_measure = 'Not exist';
end

%False alarm rate
if (TN + FP) ~= 0
    False_alarm_rate = 1 - Specificity;
else
    False_alarm_rate = 'Not exist';
end

end