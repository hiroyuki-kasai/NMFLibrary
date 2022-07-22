function [] = test_face_CBCL()
    clear; 
    clc; 
    close all;

    %% import dataset
    X = importdata('../../../data/CBCL_Face.mat'); 
    X = X';

    rank = 49; 
    
    %% perform factroization
    options.verbose = 2;
    [x, info_rnmu] = recursive_nmu(X, rank, options);
    
    %% plot
    display_graph('epoch','cost', {'R-NUM'}, [], {info_rnmu});
    
    % display results
    options.black_display = 0;
    plot_dictionnary(x.H', [], [7 7], options);

end