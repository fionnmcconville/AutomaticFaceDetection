function label = labelToDescription(num)
%Simple function that converts either an input of 0 or 1 into a
%corresponding string. 0 for non-face and 1 for face. In the case of SVM
%it's -1 for non-face and 1 for face

if num == -1 || num == 0    
    label = "Non-Face";
elseif num == 1
    label = "Face";
else
    error("Label must be either 0 or 1")
end
     
end

