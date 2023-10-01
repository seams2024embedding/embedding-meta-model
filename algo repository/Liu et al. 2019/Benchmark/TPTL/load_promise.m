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


%% all combinations of cross-project predictionss
x = load('project_id.mat');
project_id = x.res;
k = 1;
CrossProjects = cell(90,7);
for i=1:length(Projects)
    fprintf('%i Now here inside\n',i);
    target_id = project_id(i,1);
    target_project = Projects{i,2};
    source_id = find(project_id(:,1)~=target_id(1));
    source_projects = SourceProjects(project_id(:,1)~=target_id(1),:);
    for j=1:size(source_projects,1)
        CrossProjects{k,1} = i;                       
        CrossProjects{k,2} = source_id(j);          
        CrossProjects{k,3} = source_projects{j,2};     
        CrossProjects{k,4} = target_project;         
        CrossProjects{k,5} = target_project(:,11);     
        [tca_src,tca_tar]  = standard(CrossProjects{k,3},CrossProjects{k,4});
        tca_src(isinf(tca_src)) = 0 ;
        tca_tar(isinf(tca_tar)) = 0 ;
        CrossProjects{k,6} = tca_src;
        CrossProjects{k,7} = tca_tar;
        k = k + 1;
    end
end


    

