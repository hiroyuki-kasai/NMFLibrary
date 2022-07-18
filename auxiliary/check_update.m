function [update_flag, latest_version, latest_release_date, local_version] = check_update(local_version)

    if nargin < 1 
        local_version = nmflibrary_version();  
    end  

    update_flag = 0;

    site_url = 'https://raw.githubusercontent.com/hiroyuki-kasai/NMFLibrary/master/nmflibrary_version.m';
    filename_full_local = 'dl_nmflibrary_version.m';

    connection_available = check_network_connection();
    if connection_available
        websave(filename_full_local, site_url);
    
        [latest_version, latest_release_date] = dl_nmflibrary_version();
    
    
        if str2double(latest_version) > str2double(local_version)
            update_flag = 1;
            %fprintf('Latest: %s, Local: %s\n', latest_version, local_version);
        else
            %fprintf('This version is latest (%s)\n', local_version);
        end
    
        delete dl_nmflibrary_version.m
    else
        update_flag = -1;
        latest_version = [];
        latest_release_date = [];
    end

end

function available = check_network_connection()
    available = false;
    if ispc
        [~,b]=system('ping -n 1 www.google.com');
    elseif isunix
        [~,b]=system('ping -c 1 www.google.com');
    else
        error('how did you even get Matlab to work?')
    end

    n = strfind(b,'cannot resolve');
    if n > 0
    else
        available = true;
    end
end
