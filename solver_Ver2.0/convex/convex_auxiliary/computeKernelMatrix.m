function K=computeKernelMatrix(A,B,option)
% Compute the kernel matrix, K=kernel(A,B)
% A: matrix, each column is a sample
% B: matrix, each column is a sample
% option: struct, include files:
% option.kernel: string, can be 'linear','polynomial','rbf','sigmoid','ds'
% option.kernel_param
% K: the kernel matrix
% Yifeng Li, September 03, 2011

switch option.kernel
    case 'rbf'
        if isempty(option.kernel_param)
            option.kernel_param=1; % sigma
        end
%     sigma=param(1);
%     kfun= @kernelRBF; % my rbf kernel
    kfun=@kernelRBF; % fast rbf kernel from official matla
    case 'polynomial'
        if isempty(option.kernel_param)
            option.kernel_param=[1;0;2];
        end
%     Gamma=param(1);
%     Coefficient=param(2);
%     Degree=param(3);
    kfun= @kernelPoly;
    case 'linear'
        if any(any(isnan([A,B]))) % missing values
            kfun=@innerProduct;
        else
           kfun= @kernelLinear; % no missing values
        end
    case 'sigmoid'
        if isempty(option.kernel_param)
            option.kernel_param=[1;0];
        end
%         alpha=param(1);
%         beta=param(2);
        kfun=@kernelSigmoid;
    case 'ds' % dynamical systems kernel
        if size(A,3)>1
            kfun=@dynamicSystems_kernel2; % accept 3-order input  
        else
            kfun=@dynamicSystems_kernel; % 2-oder input %(D1,D2,numR,numC,rank,lambda)
        end
    otherwise
        eval(['kfunc=@',option.kernel,';']);
end

K=feval(kfun,A,B,option.kernel_param); % kernel matrix
end