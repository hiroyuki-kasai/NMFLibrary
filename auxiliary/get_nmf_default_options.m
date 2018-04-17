function options = get_nmf_default_options()

    options.max_epoch       = 100;  
    options.f_opt           = -Inf;    
    options.tol_optgap      = 1.0e-12;
    options.verbose         = 0;  
    options.store_sol       = false;
    options.batch_size      = 1;
    options.permute_on      = false;

end

