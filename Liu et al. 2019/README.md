Introduction
============
Within this archive you find the full replication package for the article "A Two-Phase Transfer Learning Model for Cross-Project Defect Prediction" by Chao Liu, Dan Yang, Xin Xia, Meng Yan, and Xiaohong Zhang for currently under review at Information and Software Technology. The aim of this replication package is to allow other researchers to replicate our results with minimal effort, as well as to provide additional results that could not be included in the article directly. 

Requirements
============
- Matlab
- Java 8
- Weka

Contents
========
- Benchmark folder with 6 models, including the proposed model TPTL and 5 baseline models (TCA+_Rnd, TCA+_All, TDS, LT, and Dycom)
- Liblinear folder with underlying classifier for defect models.
- Promise folder with 42 defect datasets stored as Attribute-Relation File Format (ARFF) files.
- Results folder with the prediction results of 6 models, scripts and outcomes of model evalutions.
- TCA+ folder with re-implemented transfer learning model TCA+
- Weka folder with the tool Weka.jar and related files.


How to use
==========
There are two ways to use this replication kit.

1. Get access to the experimental results of the TPTL model. For this, you may access the Results folder.

2. Replicate the results of the TPTL and baseline models in the Benchmark folder.

In the following, we will explain:

- how to setup your local environment for using this replication kit; and
- how the replicate the experimenal results.

Setup local Matlab workspace
---------------------------------

1. Install Matlab - any version should work, we do not use special features.

2. Open the class path file of Matlab, you may use the Matlab command "edit classpath.txt" to open the file.

3. Add the absolute path of the Weka.jar file (version 3.4.12) in Weka folder to the classpath.txt, save the file, and restart the Matlab.

Replicate the experimental results
------------------------------------

1. The proposed model TPTL and baseline models can be replicated by executing the main.m script of each subfolder in Benchmark.

2. Running a main.m script, the prediction results in F1-score and PofB20 are reported for 42 defect datasets.

License
=======
This replication package is licensed under the Apache License, Version 2.0.

 
