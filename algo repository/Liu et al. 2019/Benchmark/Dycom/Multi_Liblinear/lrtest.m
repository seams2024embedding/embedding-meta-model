function [dis,pre] = lrtest(tar,model)
    [pre,~,dis] = predict(tar(:,end),sparse(tar(:,1:end-1)),model,'-b -1 -q');
    dis = dis(:,1);
end