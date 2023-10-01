addpath('C:\Users\Noa Radin\Desktop\Semester F\lab\Liu et al. 2019 - Copy\Weka','C:\Users\Noa Radin\Desktop\Semester F\lab\Liu et al. 2019 - Copy\TCA+');

%% load 42 Promise projects
addr = 'C:\Users\Noa Radin\Desktop\Semester F\lab\Liu et al. 2019 - Copy\Promise\';
files = dir(addr);
Projects = cell(length(files)-2,1);
for i=3:length(files)
    name = [addr,files(i).name];
    Projects{i-2,1} = files(i).name;
    Projects{i-2,2} = WekaArff2Data(name);
end

%% all combinations of cross-project predictions
x = load('project_id.mat');
project_id = x.res;
CrossProjectsComb = cell(5,3);
for i=1:length(Projects)
    fprintf('%i Now here inside\n',i);
    target_id = project_id(i,1);
    target_project = Projects{i,2};
    source_id = find(project_id(:,1)~=target_id(1));
    source_projects = Projects(project_id(:,1)~=target_id(1),:);
    source_comb = [];
    for j=1:size(source_projects,1)
        source_comb = [source_comb;source_projects{j,2}];
    end
    CrossProjectsComb{i,1} = source_comb;           
    CrossProjectsComb{i,2} = target_project;           
    CrossProjectsComb{i,3} = target_project(:,11);    
end


    

