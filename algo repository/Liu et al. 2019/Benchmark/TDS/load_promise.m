addpath('Liblinear','Weka')

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
k = 1;
CrossProjects = cell(10,1);
for i=1:length(Projects)
    CrossProjects{i,1}.test.data = Projects{i,2};
    CrossProjects{i,1}.test.line = Projects{i,2}(:,11);
    t = SourceProjects(project_id(:,1)~=project_id(i,1),1:2);
    for j=1:size(t,1);
        CrossProjects{i,1}.train{j,1}.data = t{j,2};
        CrossProjects{i,1}.train{j,1}.line = t{j,2}(:,11);
    end
end


    

