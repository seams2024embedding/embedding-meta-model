javaaddpath('Weka\weka.jar', '-end')
addpath('Liblinear','Weka')

%% load 42 datasets
load_promise;

%% TDS
for i=1:length(CrossProjects)
    fprintf('%i 42\n',i);
    
    test = CrossProjects{i}.test.data;
    line = CrossProjects{i}.test.line;
    obs = test(:,end);
    
    % training data selection
    src = {};
    for j=1:length(CrossProjects{i}.train)
        src{j,1} = CrossProjects{i}.train{j}.data;
    end
    src = TDS(src,test);
    train = [];
    for j=1:length(src)
        train = [train;src{j}];
    end

    % prediction
    [pre,dis] = liblinear(train,test);
    obs2 = (obs+1)/2;
    [f1,pofb20] = WekaError(obs,pre,dis,line);
    [f1,precision,recall,accuracy] = man_weka_error(obs,pre);
    results(i,:) = [f1,precision,recall,accuracy];
end
results
load('index')
path_save = sprintf('Results\\TDS\\TDS%i.mat',res(1:1));
save(path_save,'results');
