function [] = nmflibrary_message()
    clc

    [local_version, local_release_date] = nmflibrary_version();
    [update_flag, latest_version, latest_release_date] = check_update(local_version);
    
    fprintf('\n##########################################################\n');
    fprintf('###                                                    ###\n');
    fprintf('###                Welcome to NMFLibrary               ###\n');
    fprintf('###        (version:%s, released:%s)        ###\n', local_version, local_release_date);
    
    if update_flag == 1
        fprintf('###                                                    ###\n');
        fprintf('###                                                    ###\n');    
        fprintf('###         ***  New version available !!!  ***        ###\n');
        fprintf('###        (version:%s, released:%s)        ###\n', latest_version, latest_release_date);       
        fprintf('###                                                    ###\n');    
        fprintf('###  See https://github.com/hiroyuki-kasai/NMFLibrary  ###\n');     
        fprintf('###                                                    ###\n');
    elseif update_flag == 0
        fprintf('###                                                    ###\n');            
        fprintf('###        ***  This is the latest version  ***        ###\n');   
    else
        fprintf('###                                                    ###\n');            
        fprintf('###   Update checking failed (No network connection)   ###\n');           
    end
    
    fprintf('###                                                    ###\n');
    fprintf('##########################################################\n\n\n');
end