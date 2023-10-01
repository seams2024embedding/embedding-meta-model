addpath('Weka','TCA+')

%% load 42 Promise projects
addr = 'Promise\';
files = dir(addr);
Projects = cell(length(files)-2,1);
for i=3:length(files)
    name = [addr,files(i).name];
    Projects{i-2,1} = files(i).name;
    Projects{i-2,2} = WekaArff2Data(name);
end

addr2 = 'PromiseSource\';
files2 = dir(addr2);
SourceProjects = cell(length(files2)-2,1);
for i=3:length(files2)
    name = [addr2,files2(i).name];
    SourceProjects{i-2,1} = files2(i).name;
    SourceProjects{i-2,2} = WekaArff2Data(name);
end

%% all combinations of cross-project predictions
x = load('project_id.mat');
project_id = x.res;
CrossProjectsComb = cell(10,3);
for i=1:length(Projects)
    fprintf('%i Now here inside\n',i);
    target_id = project_id(i,1);
    target_project = Projects{i,2};
    source_id = find(project_id(:,1)~=target_id(1));
    fprintf('%i Length\n',length(Projects));
    source_projects = Projects(project_id(:,1)~=target_id(1),:);
    source_comb = [];
    for j=1:size(source_projects,1)
        source_comb = [source_comb;source_projects{j,2}];
    end
    CrossProjectsComb{i,1} = source_comb;            
    CrossProjectsComb{i,2} = target_project;           
    CrossProjectsComb{i,3} = target_project(:,11);  
end


    

