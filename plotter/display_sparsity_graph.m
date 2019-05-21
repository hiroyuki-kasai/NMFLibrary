function [] = display_sparsity_graph(algorithm_list, x_list)

    figure;
    
    algorithm_num = length(algorithm_list);
    
    fs = 10;
    
    for i=1 : algorithm_num
        alg_name = algorithm_list{i};
        W = x_list{i}.W;
        H = x_list{i}.H;        
        
        m = size(W, 1);
        n = size(H, 2);    
        
        % How sparse are the basis vectors?
        cursW = (sqrt(m)-(sum(abs(W))./(sqrt(sum(W.^2))+eps))) / ((sqrt(m)-1)+eps);
        subplot(algorithm_num,2,2*(i-1)+1);
        bar(cursW);  
        xlabel_str = sprintf('W (%s)', alg_name);
        xlabel(xlabel_str, 'FontSize', fs, 'FontWeight', 'bold');
        ylabel('Sparseness', 'FontSize', fs, 'FontWeight', 'bold');    
        ylim([0 1])


        % How sparse are the coefficients
        cursH = (sqrt(n)-(sum(abs(H'))./(sqrt(sum(H'.^2))+eps))) / ((sqrt(n)-1)+eps);
        subplot(algorithm_num,2,2*i);   
        bar(cursH); 
        xlabel_str = sprintf('H (%s)', alg_name);        
        xlabel(xlabel_str, 'FontSize', fs, 'FontWeight', 'bold');
        ylabel('Sparseness', 'FontSize', fs, 'FontWeight', 'bold'); 
        ylim([0 1])
    end
end

