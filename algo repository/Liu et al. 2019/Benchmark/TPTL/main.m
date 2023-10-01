javaaddpath('Weka\weka.jar', '-end')
addpath('Liblinear','Weka','TCA+')

%% obtaining 42 defect datasets
load_promise;

%% Collecting Training Data for SPE
n              = length(CrossProjects);
cross_classify = zeros(n,4);
cross_result   = cell(n,1);
vector_f1      = zeros(n,83);
vector_pofb20  = zeros(n,83);

for i=1:length(CrossProjects)
    fprintf('%i %i\n',i,n)

    tar_id  = CrossProjects{i,1};
    src_id  = CrossProjects{i,2};
    src     = CrossProjects{i,3};
    tar     = CrossProjects{i,4};
    line    = CrossProjects{i,5};
    tca_src = CrossProjects{i,6};
    tca_tar = CrossProjects{i,7};
    obs     = tar(:,end);
    
    tca_src(isinf(tca_src)) = 0 ;
    tca_tar(isinf(tca_tar)) = 0 ;
    tca_src(isnan(tca_src))=0;
    tca_tar(isnan(tca_tar))=0;

    [tca_src,tca_tar] = tca_plus(tca_src,tca_tar);
    [pre,dis]   = liblinear(tca_src,tca_tar);
    [f1,pofb20] = WekaError(obs,pre,dis,line);
 
    cross_classify(i,:)    = [tar_id,src_id,f1,pofb20];
    cross_result{i,1}.pre  = pre;
    cross_result{i,1}.dis  = dis;
    cross_result{i,1}.line = line;
    cross_result{i,1}.obs  = obs;
 
    % calculate characteristics vector
    vector             = [tar_id,src_id,median(src(:,1:end-1)),median(tar(:,1:end-1))];
    vector_f1(i,:)     = [vector,f1];
    vector_pofb20(i,:) = [vector,pofb20];
end

%% Main Workflow of TPTL
n = length(Projects);
tptl = zeros(n,4);
for i=1:n
    fprintf('TPTL: %i %i\n',i,n);
    
    %% Phase-I: Source Project Selection Phase
    % fSPE
    test       = vector_f1(vector_f1(:,1)==i,3:end);
    train      = vector_f1(vector_f1(:,1)~=i,3:end);
    select_f1  = WekaPred(train,test,'SMOreg');
    select_f1  = select_f1.pre;
    idx_f1     = find(select_f1==max(select_f1),1);
    % cSPE
    test       = vector_pofb20(vector_pofb20(:,1)==i,3:end);
    train      = vector_pofb20(vector_pofb20(:,1)~=i,3:end);
    select_p20 = WekaPred(train,test,'SMOreg');
    select_p20 = select_p20.pre;
    idx_p20    = find(select_p20==max(select_p20),1);
    
    %% Phase-II: Transfer Learning Phase
    % note that, the transfering learning process has been done in advance,
    % and here only extract the corresponding prediction results 
    % with selected training data.
    data_f1    = cross_result(vector_f1(:,1)==i);
    pre_f1     = data_f1{idx_f1};
    f1         = pre_f1.dis;
    data_p20   = cross_result(vector_pofb20(:,1)==i);
    pre_p20    = data_p20{idx_p20};
    p20        = pre_p20.dis;
    
    %% Prediction Combination
    f1(f1>0.5) = 0.5;
    f1(f1<0.5) = 0;
    p20  = p20./2;
    dis  = f1+p20;

    pre  = dis; 
    pre(pre>=0.5) = 1; 
    pre(pre<0.5)  = -1;
    
    %% Performance Evaluation
    line        = pre_p20.line;
    obs         = pre_p20.obs;
    obs2 = (obs+1)/2;
    [f1,pofb20] = WekaError(obs,pre,dis,line);
    [f1,precision,recall,accuracy] = man_weka_error(obs,pre);
    tptlauc(i,:) = [f1,precision,recall,accuracy];
    tptl(i,:)  = [f1,precision,recall,accuracy];
end

tptl
load('index')
path_save = sprintf('Results\\TPTL\\TPTL%i.mat',res(1:1));
save(path_save,'tptl');
