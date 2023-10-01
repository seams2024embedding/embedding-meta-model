addpath('C:\Users\Noa Radin\Desktop\Semester F\lab\Liu et al. 2019 - Copy\Liblinear','C:\Users\Noa Radin\Desktop\Semester F\lab\Liu et al. 2019 - Copy\Weka','C:\Users\Noa Radin\Desktop\Semester F\lab\Liu et al. 2019 - Copy\TCA+')

%% load 42 defect projects
load_promise;

%% TCA+_All
results = [];
for i=1:5
    fprintf('%i 42\n',i);
    src  = CrossProjectsComb{i,1}; 
    tar  = CrossProjectsComb{i,2};  
    obs  = tar(:,end);        
    fprintf('%i Im here',i);
    [src,tar] = tca_plus(src,tar);
    [pre,dis] = liblinear(src,tar);
    [f1,precision,recall] = WekaError(obs,pre);
    results(i,:) = [f1,precision,recall];
end
