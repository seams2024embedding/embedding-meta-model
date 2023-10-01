addpath('Weka','TCA+');

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
CrossProjects = cell(10,3);
for i=1:length(Projects)
    target_id = project_id(i,1);
    target_project = Projects{i,2};
    source_id = find(project_id(:,1)~=target_id(1));
    source_projects = SourceProjects(project_id(:,1)~=target_id(1),:);
    CrossProjects{i,1} = source_projects(:,2);    
    CrossProjects{i,2} = target_project;          
    CrossProjects{i,3} = target_project(:,11);  
end


    

