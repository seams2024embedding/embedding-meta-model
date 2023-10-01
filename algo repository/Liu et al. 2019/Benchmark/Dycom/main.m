javaaddpath('Weka\weka.jar', '-end')
addpath('Weka','Benchmark\Dycom\Multi_Liblinear')

%% load 42 defect datasets
load_promise;

for z=1:length(CrossProjects)
    for i=1:10
        tic
        fprintf('%i 42 %i 10 \n',z,i);
        src = CrossProjects{z,1};
        tar  = CrossProjects{z,2};
        line = tar(:,11);
        obs = tar(:,end);
        
        %% Sample 10% WC data from the target project
        N    = size(tar,1);
        list = ismember(randperm(N),1:round(N*0.1));
        idx  = find(list);
        
        CC   = src;
        WC   = tar(list,:);
        Test = tar(~list,:);
        
        %% Predict on CC and WC data
        M       = size(CC,1);
        N       = size(WC,1);
        modelB  = cell(M,1);
        
        CCdis   = cell(N,1);
        CCpre   = cell(N,1);
        
        for j=1:M
            modelB{j} = lrtrain(CC{j});
            [dis,pre] = lrtest(WC,modelB{j});
            for k=1:size(WC,1)
                CCdis{k}(j,1) = dis(k);
                CCpre{k}(j,1) = pre(k);
            end
        end
        
        modelA = lrtrain(WC);
        
        %% Predict on target project
        [radis,rapre] = lrtest(tar,modelA);
        rbdis = zeros(size(tar,1),M);
        rbpre = zeros(size(tar,1),M);
        for j=1:M
            [dis,pre] = lrtest(tar,modelB{j});
            rbdis(:,j) = dis;
            rbpre(:,j) = pre;
        end
        
        wa      = 1/(M+1);
        wb      = zeros(M,1)+1/M;
        b       = zeros(M,1)+1;
        beta = 0.5;
        lr   = 0.1;
        for j=1:size(WC,1)
            x = WC(j,end);
            ydis = convert(CCdis{j});
            
            error = abs(ydis-x);
            [~,p] = max(error);
            wb(p) = wb(p)*beta;
            
            sum_wab = sum(wb) + wa;
            wb = wb./sum_wab;
            wa = wa/sum_wab;
            
            if j==1
                b = x./ydis;
            else
                b = lr.*x./ydis + (1-lr).*b;
            end
        end
        
        %% Combine prediction results
        dis = ((wb.*b)'*convert(rbdis)')' + wa*convert(radis);
        pre = dis; pre(pre>0) = 1; pre(pre<=0)=-1;
        obs2 = (obs+1)/2;
        [f1,precision,recall,accuracy] = man_weka_error(obs,pre);
        results(i,:) = [f1,precision,recall,accuracy];
    end
    results_mean(z,:) = mean(results);
    results_std(z,:) = std(results);
end

results_mean
load('index')
path_save = sprintf('Results\\Dycom\\Dycom%i.mat',res(1:1));
save(path_save,'results_mean');
