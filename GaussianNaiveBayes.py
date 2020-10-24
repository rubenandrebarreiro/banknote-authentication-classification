# -*- coding: utf-8 -*-
"""

Last update on Tue Oct 13 18:00:00 2020

@student-name: Martim Cevadinha Figueiredo
@student-email: mc.figueiredo@campus.fct.unl.pt
@student-number: 52701

@student-name: Ruben Andre Barreiro
@student-email: r.barreiro@campus.fct.unl.pt
@student-number: 42648

@degree: Master of Computer Science and Engineering (MIEI)

@college: NOVA School of Science and Technology (FCT NOVA)
@university: New University of Lisbon (UNL)

"""


# b.3) Gaussian Naïve Bayes Classifier

# Import GaussianNB (Naïve Bayes) Sub-Module,
# from SciKit-Learn Python's Library, as skl_gaussian_naive_bayes 
from sklearn.naive_bayes import GaussianNB as skl_gaussian_naive_bayes

# Import Model Selection Sub-Module, from SciKit-Learn Python's Library,
# as skl_model_selection 
from sklearn import model_selection as skl_model_selection

# Constants #1

# The Number of Folds, for Stratified K Folds, in Cross-Validation
NUM_FOLDS = 5

# -----------------------------------------------------
# \                                                   \
# \  Classifier 3) - Gaussian Naïve Bayes             \
# \___________________________________________________\


# The Function to Compute and Return the Errors for Training and Validation Sets, for the Gaussian Naïve Bayes Classifier
def compute_gaussian_naive_bayes_errors(gnb, xs, ys, train_idx, valid_idx):
    
    gnb.fit(xs[train_idx], ys[train_idx])                          # Fit the Gaussian Naïve Bayes, with the Training Set
    gnb_train_accuracy = gnb.score(xs[train_idx], ys[train_idx])   # Compute the Training Set's Accuracy (Score), for the Gaussian Naïve Bayes      
    gnb_train_error = ( 1 - gnb_train_accuracy )                   # Compute the Training Error, regarding its Accuracy (Score), for the Gaussian Naïve Bayes 
    gnb_valid_accuracy = gnb.score(xs[valid_idx], ys[valid_idx])   # Compute the Validation Set's Accuracy (Score), for the Gaussian Naïve Bayes 
    gnb_valid_error = ( 1 - gnb_valid_accuracy )                   # Compute the Validation Error, regarding its Accuracy (Score), for the Gaussian Naïve Bayes 

    return gnb_train_error, gnb_valid_error                        # Return the Training and Validation Errors, for the Gaussian Naïve Bayes


# The Function to Estimate the True/Test Error of the Testing Set, for the Gaussian Naïve Bayes Classifier
def estimate_gaussian_naive_bayes_true_test_error(xs_train, ys_train, xs_test, ys_test):
    
    gnb = skl_gaussian_naive_bayes()                          # Initialise the Gaussian Naïve Bayes Classifier
    gnb.fit(xs_train, ys_train)                                 # Fit the Gaussian Naïve Bayes, with the Testing Set
    gnb_predict_classes_xs_test = gnb.predict(xs_test)        # Predict and Classify the Values of the Testing Set, with the Gaussian Naïve Bayes Classifier TODO Confirmar
    gnb_true_test_accuracy = gnb.score(xs_test, ys_test)      # Compute the Estimated Testing Set's Accuracy (Score), for the Gaussian Naïve Bayes 
    gnb_true_test_error = ( 1 - gnb_true_test_accuracy )      # Compute the Estimated Testing Error, regarding its Accuracy (Score), for the Gaussian Naïve Bayes 
    
    num_samples_test_set = len(xs_test)                       # The Number of Samples, from the Testing Set 
    gnb_num_incorrect_predict = 0                             # The Number of Incorrect Predictions, regarding the Gaussian Naïve Bayes Classifier
    
    # For each Sample, from the Testing Set
    for current_sample_test in range(num_samples_test_set):
        
        # If the Prediction/Classification of the Class for the current Sample, of the Testing Set is different from the Real Class of the same,
        # it's considered an Real Error in Prediction/Classification, regarding the Gaussian Naïve Bayes Classifier
        if(gnb_predict_classes_xs_test[current_sample_test] != ys_test[current_sample_test] ):
            gnb_num_incorrect_predict += 1
            
    
    # Return the Predictions of the Samples, the Real Number of Incorrect Prediction and the Estimated True/Testing Error, in the Testing Set, for the Gaussian Naïve Bayes
    return gnb_predict_classes_xs_test, gnb_num_incorrect_predict, gnb_true_test_error


# The Function to Perform the Classification Process for the Gaussian Naïve Bayes Classifier
def do_gaussian_naive_bayes(ys_train_classes, xs_train_features_std, xs_test_features_std, ys_test_classes):
    
    gnb_train_error_sum = 0                                              # The sum of the Training and 
    gnb_valid_error_sum = 0                                              # Validation Errors, for Gaussian Naïve Bayes
    
    k_folds = skl_model_selection.StratifiedKFold(n_splits = NUM_FOLDS)  # The K Folds Combinations Model, for the Stratified K Folds process

        
    # The loop for all the combinations of K Folds, in the Stratified K Folds process
    for train_idx, valid_idx in k_folds.split(ys_train_classes, ys_train_classes):

        gnb = skl_gaussian_naive_bayes()                                 # Initialise the Gaussian Naïve Bayes Classifier
        
        # Compute the Training and Validation Errors, for Gaussian Naïve Bayes
        gnb_train_error, gnb_valid_error = compute_gaussian_naive_bayes_errors(gnb, xs_train_features_std, ys_train_classes, train_idx, valid_idx)
        
        gnb_train_error_sum += gnb_train_error                           # Sum the current Training and 
        gnb_valid_error_sum += gnb_valid_error                           # Validation Errors to the Sums of them
        
    
    gnb_train_error_avg_folds = ( gnb_train_error_sum / NUM_FOLDS )      # Compute the Average of the Sums 
    gnb_valid_error_avg_folds = ( gnb_valid_error_sum / NUM_FOLDS )      # of the Training and Validation Errors, by the Total Number of Folds 
    

    # Compute the Predictions of the Samples, the Real Number of Incorrect Predictions and the Estimated True/Test Error, of the Testing Set, for the Gaussian Naïve Bayes Classifier
    gnb_predict_classes_xs_test, gnb_num_incorrect_predict, estimated_gnb_true_test_error = estimate_gaussian_naive_bayes_true_test_error(xs_train_features_std, ys_train_classes, xs_test_features_std, ys_test_classes)


    # Return the Predictions of the Samples, of the Testing Set, for the Gaussian Naïve Bayes Classifier
    return gnb_train_error_avg_folds, gnb_valid_error_avg_folds, gnb_predict_classes_xs_test, gnb_num_incorrect_predict, estimated_gnb_true_test_error