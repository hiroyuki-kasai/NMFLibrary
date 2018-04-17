function [] = main_k_means()

    test_data = importdata('data/self_test.data');
    test_data = importdata('data/quiz_A.data');
    test_data = importdata('data/seed.data');
    %test_data(1,:) = [];    % delte top row
    
    ground_truth = importdata('data/self_test.ground');
    ground_truth = importdata('data/quiz_A.ground');
    ground_truth = importdata('data/seed.ground');
    %ground_truth(1,:) = []; % delte top row
    
    %
    cluster_num = length(unique(ground_truth));
    samples = size(test_data,1);
    dim = size(test_data,2);
    
    center = zeros(cluster_num,dim);
    for i=1:cluster_num
        center(i,:) = test_data(i,:);
    end

    % Do k-menas
    results = k_means(test_data, center, cluster_num);
    
    % calculate purity
    purity_result = purity(ground_truth, results);
    fprintf('Purity: %e\n', purity_result);
    
    % calculate NMI
    nmi_result = nmi(ground_truth, results);
    fprintf('NMI: %e\n', nmi_result);

end