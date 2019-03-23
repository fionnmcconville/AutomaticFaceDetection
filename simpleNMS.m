function Objects = simpleNMS(Objects,threshold)

    if exist('threshold')
        T = threshold;
    else
        T = 0.3;
    end
    
    remove_list = [];
    breakwhile = 0;
    
    if size(Objects,1) > 1
        while breakwhile == 0
            flag = 0;
            for i = 1:(size(Objects,1))            
                for j = 1:(size(Objects,1)-1)
                    if ~ismember(Objects(i,:),Objects(j+1,:),'rows')                    
                        int_area = rectint(Objects(i,1:4),Objects(j+1,1:4));
                        bb_area = Objects(j+1,3) * Objects(j+1,4);
                        if int_area / bb_area > T
                            if Objects(i,5) > Objects(j+1,5)
                                Objects(j+1,:) = [];
                                flag = 1;
                                break;
                            else
                                Objects(i,:) = [];
                                flag = 1;
                                break;
                            end
                        end
                    end
                end  
                if flag == 1
                    break;
                end
                if i == size(Objects,1)
                    breakwhile = 1;
                    flag = 1;
                    break;
                end
           end
        end
    end
end