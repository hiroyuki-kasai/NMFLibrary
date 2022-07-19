function [] = run_me_first(varargin)

    if nargin < 1
        message_flag = true;
    else
        message_flag = varargin{1};        
    end

    % Add folders to path.    
    addpath(pwd);
    
    cd solver/;
    addpath(genpath(pwd));
    cd ..;
    
    cd auxiliary/;
    addpath(genpath(pwd));
    cd ..;
    
    cd plotter/;
    addpath(genpath(pwd));
    cd ..;
    
    cd data/;
    addpath(genpath(pwd));
    cd ..;
    
    if message_flag
        nmflibrary_message();
    end

end