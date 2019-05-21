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


[version, release_date] = nmflibrary_version();
fprintf('##########################################################\n');
fprintf('###                                                    ###\n');
fprintf('###                Welcome to NMFLibrary               ###\n');
fprintf('###       (version:%s, released:%s)        ###\n', version, release_date);
fprintf('###                                                    ###\n');
fprintf('##########################################################\n');


