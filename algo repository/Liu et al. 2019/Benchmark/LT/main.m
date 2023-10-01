javaaddpath('Weka\weka.jar', '-end')
addpath('Liblinear','Weka','TCA+')

%% load 42 defect projects
load_promise;

%% LT_All
results = [];

for i=1:10
    fprintf('%i 11\n',i);
    src  = CrossProjectsComb{i,1};  
    tar  = CrossProjectsComb{i,2};  
    line = CrossProjectsComb{i,3};  
    obs  = tar(:,end);              
    
    [src,tar] = standard(src,tar);
    src(isinf(src)) = 0 ;
    tar(isinf(tar)) = 0 ;
    r = WekaClassify(src,tar,'J48');
    pre = r.pre;
    dis = r.dis;
    obs2 = (obs+1)/2;
    [f1,precision,recall,accuracy] = man_weka_error(obs,pre);
    results(i,:) = [f1,precision,recall,accuracy];
end
results
load('index')
path_save = sprintf('Results\\LT\\LT%i.mat',res(1:1));
save(path_save,'results');
