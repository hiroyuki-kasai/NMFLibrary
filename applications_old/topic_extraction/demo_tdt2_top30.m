%% Document data sets: application of NMF on the tdt2_top30 data set. 
%% NMF is able to extract meaningful topics. 
clear
close all
clc; 

load tdt2_top30 

X = X'; 

disp('*** Dataset tdt2_top30  ***'); 
fprintf('Sparsity: %2.2f%% of zero entries.\n', (1 - sum(X(:) > 0)/size(X,1)/size(X,2))*100) 
rng(2020); 
r = 20;


options.max_epoch = 100;
options.verbose = 1;
options.not_store_infos = true;
%options.alg = 'mu';
%[sol, infos] = fro_mu_nmf(fea, nClass, options);
options.alg = 'acc_hals';
[sol, ~] = als_nmf(X, r, options); 
W = sol.W;
V = sol.H;

for i = 1 : r
    [a,b] = sort(-W(:,i));
    for j = 1 : 10 % Keep the 10 words with the largest value in W(:,i) 
        Topics{j,i} = words{b(j)};
    end
end 

% Examples of topics 
% 1 : Clinton–Lewinsky scandal 
% 9 : Israeli–Palestinian conflict
% 11: stock market 
% 12: Nagano winter olympics 
% 18: sports 
% 19: The Tobacco Master Settlement Agreement (MSA) was entered in November 1998, 
% originally between the four largest United States tobacco companies 
% (Philip Morris Inc., R. J. Reynolds, Brown & Williamson and Lorillard 
%  the "original participating manufacturers", referred to as the "Majors") 
% and the attorneys general of 46 states.
% 20: 1998 Indian general election

Topics