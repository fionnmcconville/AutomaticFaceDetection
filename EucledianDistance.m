function dEuc = EucledianDistance(sample1,sample2)
%Calculates the Eucledian Distance between two samples in a dataset. Sample
%1 and sample 2 are lists which contain all of the feature values for those
%samples
distance = 0;
for i = 1:size(sample1, 2)
    distance = distance + ((sample1(i) - sample2(i))^2);
end    
dEuc = sqrt(distance);
end

