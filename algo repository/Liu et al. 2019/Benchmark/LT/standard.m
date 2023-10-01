%standard, metric based log transformation between source and target
%projects.
%   src: source project with labels
%   tar: target project with labels
%   std_src: standardized source project
%   std_tar: standardized target project
%Implemented according Camargo and Ochimizu. 2009. Towards logistic
%regression models for predicting fault-prone code across software
%projects.
function [std_src,std_tar] = standard(src,tar)
    label_src = src(:,end);
    label_tar = tar(:,end);
    src = src(:,1:end-1);
    tar = tar(:,1:end-1);

    log_src = log(src+1);
    log_tar = log(tar+1);
    m = size(src,1);
    std_tar = log_tar;
    std_src = log_src + repmat(median(log_src)-median(log_tar), m, 1);

    std_src = [std_src,label_src];
    std_tar = [std_tar,label_tar];
end