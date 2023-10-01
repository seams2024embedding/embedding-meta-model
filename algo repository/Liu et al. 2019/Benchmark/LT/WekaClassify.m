function r = WekaClassify(src,tar,type)
    Weka2Arff('train.arff',src);
    Weka2Arff('test.arff',tar);

    Source = WekaInstances('train.arff');
    Target = WekaInstances('test.arff');

    numAttr = Target.numAttributes;
    numInst = Target.numInstances;
    
    Measured = zeros(numInst,numAttr);
    for i=1:numAttr
        Measured(:,i) = Target.attributeToDoubleArray(i-1);
    end
    Measured = Measured(:,end);
    
    classifier = WekaClassifiers(type);
    Predicted = zeros(Target.numInstances,1); 
    PredictedDist = zeros(Target.numInstances,1); 

    classifier.buildClassifier(Source);
    for i = 1:Target.numInstances
        Predicted(i) = classifier.classifyInstance(Target.instance(i-1));
        tmp = classifier.distributionForInstance(Target.instance(i-1));
        PredictedDist(i) = tmp(2);
    end
    
    Measured(Measured==0)=-1;
    Predicted(Predicted==0)=-1;
    r.obs = Measured;
    r.pre = Predicted;
    r.dis = PredictedDist;
end

function D = WekaInstances(ArffFileAddress)
    loader = weka.core.converters.ArffLoader();
    loader.setFile(java.io.File(ArffFileAddress));
    D = loader.getDataSet();
    D.setClassIndex(D.numAttributes()-1);
end

function classifier = WekaClassifiers(Type)
    switch Type 
        case 'SimpleLogistic'
            classifier = weka.classifiers.functions.SimpleLogistic();
        case 'GeneticProgramming'
            classifier = weka.classifiers.functions.GeneticProgramming();
        case 'LogitBoost'
            classifier = weka.classifiers.meta.LogitBoost();
        case 'BayesNet'
            classifier = weka.classifiers.bayes.BayesNet();
        case 'RBFNetwork'
            classifier = weka.classifiers.functions.RBFNetwork();
        case 'MultilayerPerceptron'
            classifier = weka.classifiers.functions.MultilayerPerceptron();
        case 'ADTree'
            classifier = weka.classifiers.trees.ADTree();
        case 'DecisionTable'
            classifier = weka.classifiers.rules.DecisionTable();
        case 'SMO'
            classifier = weka.classifiers.functions.SMO();
        case 'SimpleKMeans'
            classifier = weka.clusterers.SimpleKMeans();
        case 'IBk'
            classifier = weka.classifiers.lazy.IBk();
        case 'J48'
            classifier = weka.classifiers.trees.J48();
    end
end

function [] = Weka2Arff(fName,data)
    data(data(:,end)~=1,end)=0;
    fid = fopen(fName,'w');
    fprintf(fid,'@relation %s\n\n',fName);
    [m,n] = size(data);
    for i=1:n-1
        fprintf(fid,'@attribute a%d numeric\n',i);
    end
    fprintf(fid,'@attribute class {0,1}\n');
    fprintf(fid,'\n@data\n');
    for i=1:m
        for j=1:n
            value = data(i,j);
%             if value == "-inf"
%                 fprintf(fid,'%d',0);
            if value == fix(value)
                fprintf(fid,'%d',value);
            else
                fprintf(fid,'%f',value);
            end
            if j==n
                fprintf(fid,'\n');
            else
                fprintf(fid,',');
            end
        end
    end
    fclose(fid);
end