function Lut = contrast_HE_lut(Iin)
%Function creates look up table produing output grey levels using histogram
%equalisation
Lut = (1:256); %initialise Lut array of size 256

cumsumlist = cumsum(imhist(Iin)); %Cumulative histogram vals for each grey level 1-255

for index = 1:numel(Lut)
    Lut(index) = max(0, uint8(256 * cumsumlist(index)/numel(Iin)) - 1); %numel(Iin) is the number of pixels in the input image
end
Lut = uint8(Lut);
end

%There's a note in the practical notes about how the index in matlab starts
%at 1 instead of 0 so you might haveto alter the transfer function
%slightly... I'm not sure how to do it so this works fine without altering
%it so just stick to this for now and perhaps take a minor hit in marks 

