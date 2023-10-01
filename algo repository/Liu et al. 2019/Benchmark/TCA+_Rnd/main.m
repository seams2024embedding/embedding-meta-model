javaaddpath('Weka\weka.jar', '-end')
addpath('Liblinear','Weka','TCA+')

%% load 42 defect projects
load_promise;

%% TCA+_Rnd
index        = cell2mat(CrossProjects(:,1:2));
results_mean = [];
results_std  = [];
results      = [];

for i=1:10  
    for j=1:3  
        
        data  = CrossProjects(index(:,1)==i,:);
        d     = data(randi(size(data,1)),:);
        fprintf('%d %d \n',d{1},d{2});
        src   = d{3};         
        tar   = d{4};
        src_name = d{5};
        tar_name = d{6};
        obs   = tar(:,end);   

        [src,tar] = tca_plus(src,tar);
        [pre,dis] = liblinear(src,tar);
        obs2 = (obs+1)/2;
        [f1,precision,recall,accuracy] = man_weka_error(obs,pre);
        results(j,:) = [f1,precision,recall,accuracy];

    end
    results_mean(i,:) = mean(results);
    results_std(i,:) = std(results);
end
results_mean
load('index')
path_save = sprintf('Results\\TCA_rnd\\TCA_rnd%i.mat',res(1:1));
save(path_save,'results_mean');
