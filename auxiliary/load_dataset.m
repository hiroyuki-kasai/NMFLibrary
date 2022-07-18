function [N, F, K, Vo, V, Ro, class_label] = load_dataset(dataset, path, rho)

    switch dataset
        case 'PIE'
            
            data = importdata('../data/PIE/PIE_pose27.mat');
            Vo_org = data.fea;
            class_label_org = data.gnd;
            Vo_org = Vo_org'; 
            %N = size(Vo_org, 2);
            N = 2800;
            N = 1000;
            Vo_org = Vo_org(:,1:N);
            class_label = class_label_org(1:N,:);
            F = size(Vo_org, 1);
            K = length(unique(class_label));

            max_gray_level = 50;
            [Vo] = normalize_image(Vo_org, max_gray_level);    

            [V, Ro] = add_outlier(rho, F, N, Vo);


        case 'AR'
            
            data = importdata('../data/AR_Face_img_28x20_downloaded.mat');
            Vo_org = data.TrainSet.X;
            class_label_org = data.TrainSet.y;
            %N = size(Vo_org, 2);
            N = 1500;
            Vo_org = Vo_org(:,1:N);
            class_label = class_label_org(:,1:N);
            F = size(Vo_org, 1);
            K = length(unique(class_label));

            max_gray_level = 50;
            [Vo] = normalize_image(Vo_org, max_gray_level);    

            [V, Ro] = add_outlier(rho, F, N, Vo);

            class_label = class_label';


        case 'COIL'
            data = importdata('../data/COIL20.mat');
            Vo_org = data.TrainSet.X;
            class_label_org = data.TrainSet.y;
            %N = size(Vo_org, 2);
            N = 700;
            Vo_org = Vo_org(:,1:N);
            class_label = class_label_org(:,1:N);
            F = size(Vo_org, 1);
            K = length(unique(class_label));

            max_gray_level = 50;
            [Vo] = normalize_image(Vo_org, max_gray_level);    

            [V, Ro] = add_outlier(rho, F, N, Vo);

            class_label = class_label';
            
        case 'CBCL'

            %Vo_org = importdata('../data/CBCL_Face.mat');            
            filename = sprintf('%s/CBCL_Face.mat', path);
            Vo_org = importdata(filename);
            N = 1000;
            Vo_org = Vo_org(:,1:N);
            F = size(Vo_org, 1);
            K = 49;

            max_gray_level = 50;
            [Vo] = normalize_image(Vo_org, max_gray_level);    

            [V, Ro] = add_outlier(rho, F, N, Vo);

            class_label = [];
            
    end

end

