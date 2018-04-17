function [d] = count_occurrence(list)

    ids = unique(list);
    d = zeros(1,length(ids));
    list_len = length(list);
    
    for i=1:list_len
        val = list(i);
        ids_idx = find(ids == val);

        d(ids_idx) = d(ids_idx) + 1;
    end
end


