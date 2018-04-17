function [ minus_h ] = cal_entropy( assignment )

    occ = count_occurrence(assignment);
    len_occ = length(occ);
    total_occ = sum(occ,2);
    h = 0;
    
    for i=1:len_occ
        p = occ(i) / total_occ;
        if p ~= 0 
            h = h + p*log(p);
        end
    end
    
    minus_h = -h;

end

