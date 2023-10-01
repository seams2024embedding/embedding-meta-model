function [model] = lrtrain(src) 
    model = train(src(:,end),sparse(src(:,1:end-1)),'-s 0 -B -1 -q');
end