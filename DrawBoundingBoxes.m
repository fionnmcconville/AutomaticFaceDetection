function [Image] = DrawBoundingBoxes(Image,Objects)
% Show the picture

Image=Image(:,:,[1 1 1]);

imshow(Image), hold on;
colours =['b';'c';'m';'y'];
% Show the detected objects
if(~isempty(Objects))
    for n=1:size(Objects,1)
        flag = 0;
        x1=Objects(n,2); y1=Objects(n,1);
        x2=x1+Objects(n,3); y2=y1+Objects(n,4);
        
        confidence = Objects(n,5)/ max(Objects(:,5));
        if confidence > 0.8
            c=1;
        elseif confidence> 0.5
            c=2;
        elseif confidence > 0.1
            c=3;
        else
            flag = 1;
            c=4;
        end
        if flag == 0
            plot([x1 x1 x2 x2 x1],[y1 y2 y2 y1 y1],colours(c));
        end
        
    end


end
