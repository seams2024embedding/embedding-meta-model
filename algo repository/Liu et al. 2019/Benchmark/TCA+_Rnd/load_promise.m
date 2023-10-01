addpath('Weka','TCA+');

%% load 42 Promise projects
addr = 'Promise\';
files = dir(addr);
Projects = cell(11-2,1);
for i=3:length(files)
    name = [addr,files(i).name];
    Projects{i-2,1} = files(i).name;
    Projects{i-2,2} = WekaArff2Data(name);
end


addr2 = 'PromiseSource\';
files2 = dir(addr2);
SourceProjects = cell(11,1);
for i=3:length(files2)
    name = [addr2,files2(i).name];
    SourceProjects{i-2,1} = files2(i).name;
    SourceProjects{i-2,2} = WekaArff2Data(name);
end


%% all combinations of cross-project predictions
x = load('project_id.mat');
project_id = x.res;
k = 1;
CrossProjects = cell(7,4);
for i=1:length(Projects)
    fprintf('%i Now here inside\n',i);
    target_id = project_id(i,1);
    target_project = Projects{i,2};
    target_project_name = Projects{i,1};
    source_id = find(project_id(:,1)~=target_id(1));
    fprintf('%i Length\n',length(Projects));
    fprintf('%i Length2\n',length(CrossProjects));
    source_projects = Projects(project_id(:,1)~=target_id(1),:);
    for j=1:size(source_projects,1)
        CrossProjects{k,1} = i;                       
        CrossProjects{k,2} = source_id(j);           
        CrossProjects{k,3} = source_projects{j,2};    
        CrossProjects{k,4} = target_project; 
        CrossProjects{k,5} =  target_project_name;
        CrossProjects{k,6} = source_projects{j,1};
        k = k + 1;
    end
end


    

