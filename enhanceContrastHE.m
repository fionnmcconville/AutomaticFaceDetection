function Iout = enhanceContrastHE(Iin)
%Enhance contrast function using look up table created by contrast_HE_lut
%function
Lut = contrast_HE_lut(Iin);
Iout = intlut(Iin,Lut);
end

