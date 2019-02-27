function [Iout, Lut] = enhanceContrastALS(Iin)
%enhanceContrastALS = enhance Contrast Automatic Linear Stretching
%Modified version of enhancedContrast function where m and c are
%automaticlly calculated without user input

% Using for loop to extract the min and max pixel values - where amount of
% pixels is >= 10 ( to reduce possible noise)
a = (1:255);
for index = 1:255
    k = find(Iin == index, 10);
    element = 0;
    if numel(Iin(k)) > 9   % using size()is unwise for 2D arrays, it gave you some headache
        num = Iin(k);
        element = num(1); 
    end
  a(index) = element;       
end
minVal = find(a, 1, 'first');
maxVal = find(a, 1, 'last');

i1 = a(minVal);    %min pixel value in image array
i2 = a(maxVal);    %max pixel value in image array 

% Both of the above have at least 10 pixels in imge that correspond to that
% value

% Look at notes in detail to see how I came up with the below formulas to
% find the m and c of this transfer function

iDiff = i2 - i1;

m = 255/iDiff;
c = -(i1 * m); % might need a minus here, not sure yet

% Now we've calculated m and c we can use the transfer function from our
% original base function
Lut = (1:256); %Create empty array indexed 1 - 256
for index = 1:numel(Lut)
    if index - 1 < -c/m
        Lut(index) = 0;
    elseif index - 1 > (255 - c)/m
        Lut(index) = 255;
    else
        Lut(index) = ((index - 1) * m) + c;
    end
end
Lut = uint8(Lut);
Iout = intlut(Iin,Lut);
end

