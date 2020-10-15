# -*- coding: utf-8 -*-
"""

Last update on Tue Oct 13 18:00:00 2020

@author: Rúben André Barreiro, 42648, MIEI

NOVA School of Science and Technology (FCT NOVA)
New University of Lisbon (UNL)

"""


# Definition of the necessary Python Libraries

# a) General Libraries:

# Import NumPy Library as np
import numpy as np

# Import PyPlot Sub-Module, from Matplotlib Python's Library as plt
import matplotlib.pyplot as plt

# Import SciKit-Learn as skl
import sklearn as skl

# Import Model Selection Sub-Module, from SciKit-Learn Python's Library,
# as skl_model_selection 
from sklearn import model_selection as skl_model_selection

# Import Brier Score Loss (Metrics) Sub-Module, from SciKit-Learn Python's Library,
# as skl_brier_score_loss
from sklearn.metrics import brier_score_loss as skl_brier_score_loss


# b) Classifiers

# b.1) Logistic Regression Classifier

# Import Logistic Regression Sub-Module, from SciKit-Learn Python's Library,
# as skl_logistic_regression 
from sklearn.linear_model import LogisticRegression as skl_logistic_regression

# b.2) Naïve Bayes Classifier, with customised KDE (Kernel Density Estimation) 
# TODO

# b.3) Gaussian Naïve Bayes Classifier

# Import Model Selection Sub-Module, from SciKit-Learn,
# as skl_gaussian_naive_bayes 
from sklearn.naive_bayes import GaussianNB as skl_gaussian_naive_bayes



# Constants

# The Number of Features
# (i.e., 4 Features, per each Banknote)
NUM_FEATURES = 4

# The Number of Classes
# (i.e., 2 Classes possible, per each Banknote, Real or Fake)
NUM_CLASSES = 2

# The Number of Folds, for Stratified K Folds, in Cross-Validation
NUM_FOLDS = 5


# The Data for Training Set
train_set_data_file = "files/data/TP1_train.tsv"

# The Data for Testing Set
test_set_data_file = "files/data/TP1_test.tsv"


# Load the Data for Training Set with NumPy function loadtxt
train_set_data_not_random = np.loadtxt(train_set_data_file, delimiter="\t")

# Load the Data for Testing Set with NumPy function loadtxt
test_set_data_not_random = np.loadtxt(test_set_data_file, delimiter="\t")



# Print of the Data for Training Set, not randomized
print("\n")
print("The Data for Training Set, not randomized:")
print(train_set_data_not_random)

# Print a new/blank line
print ("\n")


# Print of the Data for Testing Set, not randomized
print("\n")
print("The Data for Testing Set, not randomized:")
print(test_set_data_not_random)

# Print a new/blank line
print ("\n")



# Print some info about shuffling
print("Shuffling the Data for Training Set and Testing Set...")


# Shuffle the Training Set, not randomized
train_set_data_random = skl.utils.shuffle(train_set_data_not_random)

# Shuffle the Testing Set, not randomized
test_set_data_random = skl.utils.shuffle(test_set_data_not_random)



# Print of the Data for Training Set, randomized
print("\n")
print("The Data for Training Set, randomized:")
print(train_set_data_random)

# Print a new/blank line
print ("\n")


# Print of the Data for Testing Set, randomized
print("\n")
print("The Data for Testing Set, randomized:")
print(test_set_data_random)

# Print a new/blank line
print ("\n")


# Selecting the Classes of the Training Set, randomized
ys_train_classes = train_set_data_random[:,NUM_FEATURES]

# Selecting the Features of the Training Set, randomized
xs_train_features = train_set_data_random[:,0:NUM_FEATURES]


# Print the Classes of the Data for Training Set, randomized
print("\n")
print("The Classes of the Data for Training Set, randomized:")
print(ys_train_classes)

# Print a new/blank line
print ("\n")


# Print the Features of the Data for Training Set, randomized
print("\n")
print("The Features of the Data for Training Set, randomized:")
print(xs_train_features)

# Print a new/blank line
print ("\n")


# Selecting the Classes of the Testing Set, randomized
ys_test_classes = test_set_data_random[:,NUM_FEATURES]

# Selecting the Features of the Testing Set, randomized
xs_test_features = test_set_data_random[:,0:NUM_FEATURES]

# The size of the Data for Testing Set, randomized
test_set_size = len(xs_test_features)


# Print the Classes of the Data for Testing Set, randomized
print("\n")
print("The Classes of the Data for Testing Set, randomized:")
print(ys_test_classes)

# Print a new/blank line
print ("\n")


# Print the Features of the Data for Testing Set, randomized
print("\n")
print("The Features of the Data for Testing Set, randomized:")
print(xs_test_features)

# Print a new/blank line
print ("\n")


# Print the Size of the Data for Testing Set, randomized
print("\n")
print("The size of the Testing Set, randomized:")
print(test_set_size)

# Print a new/blank line
print ("\n")


# Computing the Means of the Training Set, randomized
train_means = np.mean(xs_train_features,axis=0)

# Computing the Standard Deviations of the Training Set, randomized
train_stdevs = np.std(xs_train_features,axis=0)

# Standardize the Training Set, randomized
xs_train_features_std = ( ( xs_train_features - train_means ) / train_stdevs )


# Print the Standardized Features of the Data for Training Set, randomized
print("\n")
print("The Standardized Features of the Data for Training Set, randomized:")
print(xs_train_features_std)

# Print a new/blank line
print ("\n")


# Computing the Means of the Testing Set, randomized
test_means = np.mean(xs_train_features,axis=0)

# Computing the Standard Deviations of the Testing Set, randomized
test_stdevs = np.std(xs_train_features,axis=0)




# -----------------------------------------------------
# \                                                   \
# \  Classifier 1) - Logistic Regression,             \
# \  varying its Regularization C parameter           \
# \___________________________________________________\

# Function to Compute and Return  the Errors for Training and Validation Sets 
def compute_logistic_regression_errors(xs, ys, train_idx, valid_idx, c_param_value, score_type = 'brier_score'):
    
    # Initialise the Logistic Regression,
    # from the Linear Model of the SciKit-Learn
    logistic_regression = skl_logistic_regression(C = c_param_value, tol = 1e-10)
    
    # Fit the Logistic Regression 
    logistic_regression.fit(xs[train_idx,:NUM_FEATURES], ys[train_idx])
    
    # Compute the prediction probabilities of some Features,
    # belonging to a certain Class, due to the 
    ys_logistic_regression_prediction_probabilities = logistic_regression.predict_proba(xs[:,:NUM_FEATURES])[:,1]
    
    
    # Compute the Training and Validation Errors, based on a certain type of Scoring
    if(score_type == 'brier_score'):
    
        # Compute the Training Error, related to its Brier Score
        train_error = skl_brier_score_loss(ys[train_idx], ys_logistic_regression_prediction_probabilities[train_idx])

        # Compute the Validation Error, related to its Brier Score        
        valid_error = skl_brier_score_loss(ys[valid_idx], ys_logistic_regression_prediction_probabilities[valid_idx])

    if(score_type == 'logistic_regression'):
        
        # Compute the Training Set's Accuracy (Score), for the Logistic Regression
        accuracy_train = logistic_regression.score(xs[train_idx], ys[train_idx])
    
        # Compute the Validation Set's Accuracy (Score), for the Logistic Regression    
        accuracy_valid = logistic_regression.score(xs[valid_idx], ys[valid_idx])
        
        # Compute the Training Error, regarding its Accuracy (Score)
        train_error = ( 1 - accuracy_train )
        
        # Compute the Validation Error, regarding its Accuracy (Score)
        valid_error = ( 1 - accuracy_valid )

        
    # Return the Training and Validation Errors, for the Logistic Regression
    return train_error, valid_error


def plot_train_valid_error_logistic_regression(train_error_values, valid_error_values):
    
    plt.figure(figsize=(8, 8), frameon=True)
    plt.plot(train_error_values[:,0], train_error_values[:,1],'-', color="blue")
    plt.plot(valid_error_values[:,0], valid_error_values[:,1],'-', color="red")
    
    plt.axis([np.log(1e-2),np.log(1e12),min(valid_error_values[:,1]),max(valid_error_values[:,1])])
    plt.title('Logistic Regression\n\nTraining Error (Blue) / Cross-Validation Error (Red)')
    plt.savefig('files/imgs/LR.png', dpi=600)
    plt.show()
    plt.close()

def estimate_logistic_regression_true_test_error(xs_train, ys_train, xs_test, ys_test, best_c_param_value=1e12, score_type = 'brier_score'):
    
    logistic_regression = skl_logistic_regression(C=best_c_param_value, tol=1e-10)
   
    logistic_regression.fit(xs_train[:,:NUM_FEATURES], ys_train)
    ys_logistic_regression_prediction_probabilities = logistic_regression.predict_proba(xs_test[:,:NUM_FEATURES])[:,1]
    
    
    # Estimate the Testing Error, based on a certain type of Scoring
    # 1) Brier Scoring
    if(score_type == 'brier_score'):
    
        # Estimate the Testing Error, related to its Brier Score
        estimated_true_test_error = skl_brier_score_loss(ys_test, ys_logistic_regression_prediction_probabilities)

    # 2) Logistic Regression Scoring
    if(score_type == 'logistic_regression_score'):
        
        # Compute the Training Set's Accuracy (Score), for the Logistic Regression
        estimated_accuracy_test = logistic_regression.score(xs_test, ys_test)
    
        # Compute the Training Error, regarding its Accuracy (Score)
        estimated_true_test_error = ( 1 - estimated_accuracy_test )
        
    
    return estimated_true_test_error


def do_logistic_regression():
    
    print("-----------------------------------------------------------------")
    print("1) Starting the Logistic Regression Classifier...")
    print("-----------------------------------------------------------------")    
    
    # The K Folds, for the Stratified K Folds
    k_folds = skl_model_selection.StratifiedKFold(n_splits = NUM_FOLDS)
    
    
    logistic_regression_best_valid_error_avg_folds = 1e10
    
    logistic_regression_best_c_param_value = 1e10
    
    
    initial_exp_factor = 0
    final_exp_factor = 15
    
    initial_c_param_value = 1e-2
    
    
    logistic_regression_train_error_values = np.zeros((15,2))
    logistic_regression_valid_error_values = np.zeros((15,2))
    
    
    for current_exp_factor in range(initial_exp_factor, final_exp_factor):
    
        logistic_regression_train_error_sum = 0
        logistic_regression_valid_error_sum = 0
        
        current_c_param_value = ( initial_c_param_value * 10**(current_exp_factor) )
        
        print("Trying the Regularization Parameter C = {},\nfor Logistic Regression...".format(current_c_param_value))
        print("\n")
        
        for train_idx, valid_idx in k_folds.split(ys_train_classes,ys_train_classes):
            
            logistic_regression_train_error, logistic_regression_valid_error = compute_logistic_regression_errors(xs_train_features_std, ys_train_classes, train_idx, valid_idx, current_c_param_value, 'brier_score')
            
            logistic_regression_train_error_sum += logistic_regression_train_error
            logistic_regression_valid_error_sum += logistic_regression_valid_error
            
        logistic_regression_train_error_avg_folds = ( logistic_regression_train_error_sum / NUM_FOLDS )
        logistic_regression_valid_error_avg_folds = ( logistic_regression_valid_error_sum / NUM_FOLDS )
        
        
        print("Current Value for Regularization C = {} :".format(current_c_param_value))
        print("- Training Error = {} ; - Validation Error = {}".format(logistic_regression_train_error_avg_folds, logistic_regression_valid_error_avg_folds))
        print("\n")
        
        if(logistic_regression_best_valid_error_avg_folds > logistic_regression_valid_error_avg_folds):
            logistic_regression_best_valid_error_avg_folds = logistic_regression_valid_error_avg_folds
            logistic_regression_best_c_param_value = current_c_param_value
            
    
        print("Storing the Training and Validation Errors, for the future Plot of Errors...")
        print("\n")
        
        logistic_regression_train_error_values[current_exp_factor, 0] = np.log(current_c_param_value)
        logistic_regression_train_error_values[current_exp_factor, 1] = logistic_regression_train_error_avg_folds
        
        logistic_regression_valid_error_values[current_exp_factor, 0] = np.log(current_c_param_value)
        logistic_regression_valid_error_values[current_exp_factor, 1] = logistic_regression_valid_error_avg_folds
        
                
    print("\n")
    print("Best Value for Regularization C = {} :".format(logistic_regression_best_c_param_value))
    print("- Best Validation Error = {}".format(logistic_regression_best_valid_error_avg_folds))
    print("\n")


    plot_train_valid_error_logistic_regression(logistic_regression_train_error_values, logistic_regression_valid_error_values)
    
    logistic_regression_estimated_true_test_error = estimate_logistic_regression_true_test_error(xs_train_features, ys_train_classes, xs_test_features, ys_test_classes, logistic_regression_best_c_param_value, 'brier_score')    


    print("\n")
    print("- Estimated True/Test Error = {}".format(logistic_regression_estimated_true_test_error))
    

# -----------------------------------------------------
# \                                                   \
# \  Classifier 2) - Naïve Bayes,                     \
# \  with customised KDE (Kernel Density Estimation), \
# \  varying its Bandwidth Regularization parameter   \
# \___________________________________________________\


def do_naive_bayes():
    
    print()
    


# -----------------------------------------------------
# \                                                   \
# \  Classifier 3) - Gaussian Naïve Bayes             \
# \___________________________________________________\


def compute_gaussian_naive_bayes_errors(gaussian_naive_bayes, xs, ys, train_idx, valid_idx):
    
    gaussian_naive_bayes.fit(xs[train_idx], ys[train_idx])
    
    
    gaussian_naive_bayes_train_accuracy = gaussian_naive_bayes.score(xs[train_idx], ys[train_idx])
    
    gaussian_naive_bayes_train_error = ( 1 - gaussian_naive_bayes_train_accuracy )


    gaussian_naive_bayes_valid_accuracy = gaussian_naive_bayes.score(xs[valid_idx], ys[valid_idx])
    
    gaussian_naive_bayes_valid_error = ( 1 - gaussian_naive_bayes_valid_accuracy )


    return gaussian_naive_bayes_train_error, gaussian_naive_bayes_valid_error


def estimate_gaussian_naive_bayes_true_test_error(xs_test, ys_test):
    
    gaussian_naive_bayes = skl_gaussian_naive_bayes()
    
    gaussian_naive_bayes.fit(xs_test, ys_test)
    
    gaussian_naive_bayes_classification = gaussian_naive_bayes.predict(xs_test)
    

    gaussian_naive_bayes_true_test_accuracy = gaussian_naive_bayes.score(xs_test, ys_test)
    
    gaussian_naive_bayes_true_test_error = ( 1 - gaussian_naive_bayes_true_test_accuracy )

    
    return gaussian_naive_bayes_true_test_error


def do_gaussian_naive_bayes():
    
    print("-----------------------------------------------------------------")
    print("3) Starting the Gaussian Naïve Bayes Classifier...")
    print("-----------------------------------------------------------------")
    
    
    gaussian_naive_bayes_train_error_sum = 0
    gaussian_naive_bayes_valid_error_sum = 0
    
    # The K Folds, for the Stratified K Folds
    k_folds = skl_model_selection.StratifiedKFold(n_splits = NUM_FOLDS)
    
    
    for train_idx, valid_idx in k_folds.split(ys_train_classes,ys_train_classes):
            
        gaussian_naive_bayes = skl_gaussian_naive_bayes()
        
        gaussian_naive_bayes_train_error, gaussian_naive_bayes_valid_error = compute_gaussian_naive_bayes_errors(gaussian_naive_bayes, xs_train_features_std, ys_train_classes, train_idx, valid_idx)
        
        gaussian_naive_bayes_train_error_sum += gaussian_naive_bayes_train_error
        gaussian_naive_bayes_valid_error_sum += gaussian_naive_bayes_valid_error
        
        
    gaussian_naive_bayes_train_error_avg_folds = ( gaussian_naive_bayes_train_error_sum / NUM_FOLDS )
    gaussian_naive_bayes_valid_error_avg_folds = ( gaussian_naive_bayes_valid_error_sum / NUM_FOLDS )
    
    print("\n")
    print("- Training Error = {}".format(gaussian_naive_bayes_train_error_avg_folds))
    print("- Validation Error = {}".format(gaussian_naive_bayes_valid_error_avg_folds))
    print("\n")
    
    estimated_gaussian_naive_bayes_true_test_error = estimate_gaussian_naive_bayes_true_test_error(xs_test_features, ys_test_classes)

    print("\n")
    print("- Estimated True/Test Error = {}".format(estimated_gaussian_naive_bayes_true_test_error))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    

# ---- Run the 3 Classifiers: ------

# 1) Logistic Regression,
#    varying its Regularization C parameter    
do_logistic_regression()

print("\n\n")


# 2) Naïve Bayes, with customised KDE (Kernel Density Estimation),
#    varying its Bandwidth Regularization parameter
do_naive_bayes()

print("\n\n")


# 3) Gaussian Naïve Bayes
do_gaussian_naive_bayes()

print("\n\n")