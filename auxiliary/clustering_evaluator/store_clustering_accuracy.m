function [infos] = store_clustering_accuracy(H, gnd, classnum, infos, eval_num, iter)

%   gnd: ground truth label (Nx1)

    accuracy = eval_clustering_accuracy(H, gnd, classnum, eval_num);
    
    if ~iter
        infos.clustering_acc = accuracy;
    else    
        infos.clustering_acc = [infos.clustering_acc accuracy];
    end
end

