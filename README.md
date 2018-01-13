# clickstreamMining
Clickstream Mining with Decision Trees

The project is based on a task posed in KDD Cup 2000. It involves mining click-stream data collected from Gazelle.com, which sells legware products. Our task is to determine: Given a set of page views, will the visitor view another page on the site or will he leave?

The data set given to you is in a different form than the original. In particular it has discretized numerical values obtained by partitioning them into 5 intervals of equal frequency. This way, we get a data set where for each feature, we have a finite set of values. These values are mapped to integers, so that the data is easier for you to handle. 

You have 5 files in .csv format

trainfeat.csv: Contains 40000 examples, each with 274 features in the form of a 40000 x 274 matrix.
trainlabs.csv: Contains the labels (class) for each training example (did the visitor view another page?)
testfeat.csv: Contains 25000 examples, each with 274 features in the form of a 25000 x 274 matrix.
testlabs.csv: Contains the labels (class) for each testing example.
featnames.csv: Contains the "names" of the features. These might useful if you need to know what the features represent.

The objective is to implement the ID3 decision tree learner on Python. Our program uses the chi-squared split stopping criterion with the p-value threshold given as a parameter. Use your implementation with the threshold for the criterion set to 0.05, 0.01 and 1. Remember, 1 corresponds to growing the full tree.
