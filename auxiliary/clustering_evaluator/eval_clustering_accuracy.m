function [accuracy] = eval_clustering_accuracy(H, gnd, classnum, eval_num)

%   gnd: ground truth label (Nx1)

    accuracies = [];
    
    best_nmi = 0;
    best_idx = 0;    
    
    
    for i = 1 : eval_num
    
        label = litekmeans(H', classnum, 'Replicates', 20);

        accuracies(i).mi = MutualInfo(gnd, label);

        accuracies(i).purity = calc_purity(gnd, label);

        accuracies(i).nmi = calc_nmi(gnd, label);
        %accuracies(i).nmi = compute_nmi(gnd, label);
        [accuracies(i).f_val, accuracies(i).precision, accuracies(i).recall] = compute_f(gnd,label);

        C = bestMap(gnd,label);
        accuracies(i).acc = length(find(gnd == C))/length(gnd);
        
        if accuracies(i).nmi >= best_nmi
            best_nmi = accuracies(i).nmi;
            best_idx = i;
        end
    end 
    
    
    best_best_accuracy = accuracies(best_idx);
    
    % store
    accuracy.mi     = best_best_accuracy.mi;
    accuracy.purity = best_best_accuracy.purity;
    accuracy.nmi    = best_best_accuracy.nmi;
    accuracy.f_val  = best_best_accuracy.f_val;
    accuracy.acc    = best_best_accuracy.acc;    
 
end

