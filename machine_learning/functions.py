# Functions for performing operations & computations

# Import libraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from scipy import interp
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFECV

from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix


def get_feats(string, df): 
    """ 
        Purpose: get feature names with specified string
        
        Args: 
          string: string for identifying features 
          df: dataframe
          
        Returns: list of feature names with specified string
        
    """
    feats = list(filter(lambda col: string in col, df.columns))
    return feats

def plot_feats(feats, bins, df): 
    """ 
        Purpose: visualize distribution of features 
        
        Args: 
         feats: list of features to be plotted
         bins: array of start, end points of bins
         df: dataframe
         
        Output: distribution of features by labels
        
    """ 
    
    plots = ['overall', 'label 0', 'label 1']
    fig, ax = plt.subplots(nrows = len(feats), ncols = len(plots), figsize = (15,15))
    
    for i, feat in enumerate(feats): 
        for j, plot in enumerate(plots):
            if j == 0:
                ax[i,j].hist(df[feat], bins = bins, density = True, edgecolor = 'black')
            elif j == 1:
                ax[i,j].hist(df[feat][df['repeat_attempt_within_oneyear'] == 0], bins = bins, 
                                      density = True, edgecolor = 'black')
            else:
                ax[i,j].hist(df[feat][df['repeat_attempt_within_oneyear'] == 1], bins = bins, 
                                      density = True, edgecolor = 'black')
            ax[i,j].set(title = 'Distribution of {feat} for {plot}'.format(feat = feat, plot = plot), 
                        xlabel = feat, ylabel = 'Proportion', ylim = [0,1])
        
    plt.show()
    
def get_distribution(feats, df): 
    """
        Purpose: get frequencies, proportions of features 
        
        Args: 
          feats: list of feature names 
          df: dataframe
        
        Prints: frequencies, proportions of features 
        
    """
    for feat in feats:
        print(df[feat].value_counts())
        print(round(df[feat].value_counts(normalize = True), 3))
        print('\n') 
        
def get_metrics(pipe, x_test, y_test):
    """
        Purpose: compute model performance metrics 
        
        Args: 
          pipe: a pipeline
          x_test: test set of features
          y_test: test set of labels
        
        Prints: metric scores     
        
    """
    y_pred = pipe.predict(x_test)
    print('accuracy: {a}'.format(a = round(pipe.score(x_test, y_test), 3)))
    print('recall: {r}'.format(r = round(recall_score(y_test, y_pred), 3)))
    print('precision: {p}'.format(p = round(precision_score(y_test, y_pred), 3)))
    print('f1: {f}'.format(f = round(f1_score(y_test, y_pred), 3)))
    print('roc auc: {r}'.format(r = round(roc_auc_score(y_test, y_pred), 3)))
    print('classification report:')
    print(classification_report(y_test, y_pred))
    print('confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    
def kf_cv(pipe, x_train, y_train, k_folds = 5):
    """ 
       Purpose: implements k-fold cv for multiple metrics and plots roc curve
       
       Args: 
         pipe: a pipeline
         x_train: training set of features
         y_train: training set of labels
         k_folds: number of folds
       
       Plots: roc curve per fold, mean roc curve for all folds
       
       Returns: mean of metrics for all folds
       
    """
    kfold = KFold(n_splits = k_folds)
    scores = np.empty((3, k_folds))
    tprs = []
    mean_fpr = np.linspace(0,1,100)
    
    # iterate through each fold
    for k, (train_indices, test_indices) in enumerate(kfold.split(x_train, y_train)):
        pipe.fit(x_train[train_indices], y_train[train_indices])
        y_train_pred = pipe.predict(x_train[test_indices])
        y_train_prob = pipe.predict_proba(x_train[test_indices])
        
        # compute recall, precision, f1 
        scores[:,k] = np.array([recall_score(y_train[test_indices], y_train_pred), # recall
                                precision_score(y_train[test_indices], y_train_pred), # precision
                                f1_score(y_train[test_indices], y_train_pred)]) # f1 
        
        # plot roc curve
        fpr, tpr, _ = roc_curve(y_train[test_indices], y_train_prob[:,1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label = 'Fold {k}: {roc_auc}'.format(k = k+1, roc_auc = round(roc_auc,3)))
    
    # plot mean roc curve 
    mean_tpr = np.mean(tprs, axis = 0)
    mean_tpr[-1] = 1
    mean_auc = auc(mean_fpr, mean_tpr) # mean roc auc 
    plt.plot(mean_fpr, mean_tpr, label = 'Mean ROC AUC: {mean_auc}'.format(mean_auc = round(mean_auc,3)))
    plt.title('ROC curve for {k_folds}-fold cv'.format(k_folds = k_folds))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc = 'lower right')
    plt.show()
    
    # return mean recall, precision, f1, roc auc for all folds
    return np.append(scores.mean(axis = 1).ravel(), mean_auc)

def run_pipe(pipes, x_train, y_train, names): 
    """ 
        Purpose: executes pipelines with k-fold cv
        
        Args: 
           pipes: list of pipelines 
           x_train: training set of features
           y_train: training set of labels
           names: list of classifier names
           
        Returns: a dictionary with pipe:metrics pairs 
        
    """
    pipe_scores = {}
    for i, p in enumerate(pipes):
        print(names[i])
        pipe_scores[names[i]] = kf_cv(p, x_train.values, y_train.values)
    return pipe_scores

def format_results(d): 
    """
       Purpose: sort and format results
       
       Args: 
         d: a dictionary of model:results pairs
         
       Returns: the dictionary sorted by highest to lowest recall score
       
    """
    sorted_results = OrderedDict(sorted(d.items(), key = lambda x: -x[1][0]))
    for k, v in sorted_results.items():
        print(k)
        print('recall: {r}\nprecision: {p}\nf1: {f}\nroc auc: {a}'.format(r = round(v[0],3), 
                                                                          p = round(v[1],3), 
                                                                          f = round(v[2],3), 
                                                                          a = round(v[3],3)))
        print('\n')
    return sorted_results


        
 
    