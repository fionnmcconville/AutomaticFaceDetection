function [Accuracy] = CV_Evaluation(trainLabs_fold, testLabs_fold, accuracy, tp, fp, tn, fn, recall, precision, sensitivity, specificity, f_measure, false_alarm_rate, fold_size)

%% Function to calculate the evaluation elements' average value over the cross validation, 
% and show them in figures

TP = 0;
FP = 0;
TN = 0;
FN = 0;
Recall = 0;
Precision = 0;
Specificity = 0;
Sensitivity = 0; 
F_measure = 0;
False_alarm_rate = 0;

% Table elements record accuracy in every Features sets(with different number
%of f)
CV_Sequence = {};
CV_Accuracy =  {};
Accuracy = 0;


%Record number of useful Precision, recall, specificity, etc
count_recall = 0;
count_precision = 0;
count_sensitivity = 0;
count_specificity = 0;
count_f_measure = 0;
count_false_alarm_rate = 0;

for j = 1:fold_size(1,1)
    Accuracy = Accuracy + accuracy(:,j);
    CV_Sequence = [CV_Sequence; j];
    CV_Accuracy = [CV_Accuracy; accuracy(:,j)];
    TP = TP + tp(:,j);
    FP = FP + fp(:,j);
    TN = TN + tn(:,j);
    FN = FN + fn(:,j);
    if recall(:,j) ~= Inf
        Recall = Recall + recall(:,j);
        count_recall = count_recall + 1;
    end
    if precision(:,j) ~= Inf
        Precision = Precision + precision(:,j);
        count_precision = count_precision + 1;
    end
    if sensitivity(:,j) ~= Inf
        Sensitivity = Sensitivity + sensitivity(:,j);
        count_sensitivity = count_sensitivity + 1;
    end
    if specificity(:,j) ~= Inf
       Specificity = Specificity + specificity(:,j);
       count_specificity = count_specificity + 1;
    end
    if f_measure(:,j) ~= Inf     
       F_measure = F_measure + f_measure(:,j);
       count_f_measure = count_f_measure + 1;
    end
    if false_alarm_rate(:,j) ~= Inf
       False_alarm_rate = False_alarm_rate + false_alarm_rate(:,j);
       count_false_alarm_rate = count_false_alarm_rate + 1;
    end
end

%Calculate the average of each evaluation variable
Accuracy = Accuracy / double(fold_size(1,1));
CV_Sequence = [CV_Sequence; 'Average'];
CV_Accuracy = [CV_Accuracy; Accuracy];
TP = int16(TP / double(fold_size(1,1)));
FP = int16(FP / double(fold_size(1,1)));
TN = int16(TN / double(fold_size(1,1)));
FN = int16(FN / double(fold_size(1,1)));
Recall = Recall / count_recall;
Precision = Precision / count_precision;
Sensitivity = Sensitivity / count_sensitivity;
Specificity = Specificity / count_specificity;
F_measure = F_measure / count_f_measure;
False_alarm_rate = False_alarm_rate / count_false_alarm_rate;

%Build a table recording accuracy in every Features sets(with different number
%of f)
CV_AccuracyTable = table(CV_Sequence, CV_Accuracy);

%Show Accuracy table  in a figure
Accuracy_show = uifigure;
uitable(Accuracy_show, 'Data', CV_AccuracyTable);

%Build a confusion matrix
Confusion_Matrix = {'Actual_Face'; 'Actual_NonFace'};
Predict_Face = {TP; FP};
Predict_NonFace =  {FN; TN};
Confusion_Table = table(Confusion_Matrix, Predict_Face, Predict_NonFace);

%Show confusion matrix in a figure
Confusion_matrix_show = uifigure;
uitable(Confusion_matrix_show, 'Data', Confusion_Table);

%Record test images size and precision, recall, specificity, etc in a matrix
Evaluation_Name = {'Accuracy';'train size every time '; 'test size every time'; 'Recall'; 'Precision'; 'Specificity';'Sensitivity'; 'F-measure'; 'False alarm rate'};
testLabs_fold_size = size(testLabs_fold);
trainLabs_fold_size = size(trainLabs_fold);
Values = {Accuracy; trainLabs_fold_size(1,1); testLabs_fold_size(1,1); Recall; Precision; Specificity; Sensitivity; F_measure; False_alarm_rate};
Precision_Sensitivity = table(Evaluation_Name, Values);

%Show test images size and precision, recall, specificity, etc in a figure
Precision_Sensitivity_show = uifigure;
uitable(Precision_Sensitivity_show, 'Data', Precision_Sensitivity);


end