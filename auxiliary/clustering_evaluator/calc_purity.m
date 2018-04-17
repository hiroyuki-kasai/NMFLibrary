function [ purity_val ] = calc_purity(groundtruthAssignment, algorithmAssignment)

    matching = 0;
    ids = unique(algorithmAssignment);
    ids_len = length(ids);
    
    for i=1:ids_len
        ids_val = ids(i);
        indices = algorithmAssignment == ids_val;
        cluster = groundtruthAssignment(indices);
        occ = count_occurrence(cluster);
        matching = matching + max(occ);
    end
    
    purity_val =  matching / length(groundtruthAssignment);
end

