function [ nmi_val ] = calc_nmi( groundtruthAssignment, algorithmAssignment )

    nmi_val = 0;
    
    h_c = cal_entropy(algorithmAssignment); % Entropy of clustering C
    h_t = cal_entropy(groundtruthAssignment); % Entropy of partitioning T
    
    % compute Mutual information
    occ_c = count_occurrence(algorithmAssignment); % get occurrence: for the probability of cluster C_id
    n_c = sum(occ_c,2); % total # of cluster C_id
    occ_t = count_occurrence(groundtruthAssignment); % get occurrence: for the probability of cluster T_id
    n_t = sum(occ_t,2); % total # of cluster T_id
    ids_c = unique(algorithmAssignment);
    ids_t = unique(groundtruthAssignment);
    
    % create cp dictonary (cartesian product for all possible id combination)
    cp_len = length(ids_c) * length(ids_t);
    cp = zeros(cp_len,2);
    cp_idx = 1;
    for i=1:length(ids_c)
      for j=1:length(ids_t)
          cp(cp_idx,:) = [ids_c(i), ids_t(j)];
          cp_idx = cp_idx + 1;
      end
    end
    
    % check numbers
    if n_c ~= n_t
        fprintf('Error');
        return;
    else
       total_samples = n_c; 
    end
    
    % count occurrence in cp dictonary
    cp_count = zeros(1,cp_len);
    for i=1:total_samples
        for cp_idx=1:cp_len
            if cp(cp_idx,1) == algorithmAssignment(i) && cp(cp_idx,2) == groundtruthAssignment(i)
                cp_count(cp_idx) = cp_count(cp_idx) + 1;
            end
        end
    end
    
    % calculate nmi
    mi = 0; % mutual information
    for cp_idx=1:cp_len
        if cp_count(cp_idx) ~= 0
            c_idx = cp(cp_idx,1);
            occ_c_idx = ids_c == c_idx;
            occ_c_count = occ_c(occ_c_idx);
            
            
            t_idx = cp(cp_idx,2);
            occ_t_idx = ids_t == t_idx;
            occ_t_count = occ_t(occ_t_idx);
            
            mi = mi + (cp_count(cp_idx)/n_c) * log( (cp_count(cp_idx)/n_c) / ((occ_c_count/n_c)*(occ_t_count/n_t)) );
        end
    end
    
    nmi_val = mi / sqrt(h_c*h_t);
    
end

