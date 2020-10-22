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

# Import NumPy Python's Library as np
import numpy as np

# Import Logistic Regression Sub-Module, from SciKit-Learn Python's Library,
# as skl_logistic_regression 
from sklearn.linear_model import LogisticRegression as skl_logistic_regression

# Import Model Selection Sub-Module, from SciKit-Learn Python's Library,
# as skl_model_selection 
from sklearn import model_selection as skl_model_selection

# Import Brier Score Loss (Metrics) Sub-Module,
# from SciKit-Learn Python's Library, as skl_brier_score_loss
from sklearn.metrics import brier_score_loss as skl_brier_score_loss

# Import PyPlot Sub-Module, from Matplotlib Python's Library as plt
import matplotlib.pyplot as plt

# The Number of Folds, for Stratified K Folds, in Cross-Validation
NUM_FOLDS = 5

# The Number of Steps/Variations for ajusting the C Regularization parameter,
# for the Logistic Regression
NUM_STEPS_C_REGULARIZATION_LOGISTIC_REGRESSION = 15



# -----------------------------------------------------
# \                                                   \
# \  Classifier 1) - Logistic Regression,             \
# \  varying its Regularization C parameter           \
# \___________________________________________________\

    
# The Function to Compute and Return the Errors for Training and Validation Sets, for the Logistic Regression Classifier
def compute_logReg_errors(xs, ys, train_idx, valid_idx, c_param_value, num_features, score_type = 'brier_score'):
    
    # Initialise the Logistic Regression, from the Linear Model of the SciKit-Learn
    logReg = skl_logistic_regression(C = c_param_value, tol = 1e-10)
    
    # Fit the Logistic Regression 
    logReg.fit(xs[train_idx,:num_features], ys[train_idx])
    
    # Compute the prediction probabilities of some Features, belonging to a certain Class, due to the 
    ys_logReg_predict_prob = logReg.predict_proba(xs[:,:num_features])[:,1]
    
    
    # Compute the Training and Validation Errors, based on a certain type of Scoring:
    # 1) Based on Brier Score
    
    if(score_type == 'brier_score'):
        logReg_train_error = skl_brier_score_loss(ys[train_idx], ys_logReg_predict_prob[train_idx])     # Compute the Training Error, related to its Brier Score
        logReg_valid_error = skl_brier_score_loss(ys[valid_idx], ys_logReg_predict_prob[valid_idx])     # Compute the Validation Error, related to its Brier Score   

    # 2) Based on Logistic Regression Score
    
    if(score_type == 'logistic_regression_score'):
        logReg_accuracy_train = logReg.score(xs[train_idx], ys[train_idx])  # Compute the Training Set's Accuracy (Score), for the Logistic Regression
        logReg_accuracy_valid = logReg.score(xs[valid_idx], ys[valid_idx])  # Compute the Validation Set's Accuracy (Score), for the Logistic Regression
        logReg_train_error = ( 1 - logReg_accuracy_train )                  # Compute the Training Error, regarding its Accuracy (Score)
        logReg_valid_error = ( 1 - logReg_accuracy_valid )                  # Compute the Validation Error, regarding its Accuracy (Score)

        
    # Return the Training and Validation Errors, for the Logistic Regression
    return logReg_train_error, logReg_valid_error


# The Function to Plot the Training and Validation, for the Logistic Regression
def plot_train_valid_error_logistic_regression(train_error_values, valid_error_values):
    
    # Initialise the Plot
    plt.figure(figsize=(8, 8), frameon=True)

    # Set the line representing the continuous values,
    # for the Functions of the Training and Validation Errors
    plt.plot(train_error_values[:,0], train_error_values[:,1],'-', color="blue")
    plt.plot(valid_error_values[:,0], valid_error_values[:,1],'-', color="red")
    
    # Set the axis for the Plot
    plt.axis([min(valid_error_values[:,0]), max(valid_error_values[:,0]), min(valid_error_values[:,1]), max(valid_error_values[:,1])])
    
    # Set the laber for the X axis of the Plot
    plt.xlabel("log(C)")
    
    # Set the laber for the Y axis of the Plot
    plt.ylabel("Training/Validation Errors")
    
    # Set the Title of the Plot
    plt.title('Logistic Regression, varying the C parameter\n\nTraining Error (Blue) / Cross-Validation Error (Red)')
    
    # Save the Plot, as a figure/image
    plt.savefig('files/imgs/LR.png', dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()


# The Function to Estimate the True/Test Error of the Testing Set,
# for the Logistic Regression Classifier
def estimate_logReg_true_test_error(xs_train, ys_train, xs_test, ys_test, num_features, best_c_param_value=1e12, score_type = 'brier_score'):
    
    logReg = skl_logistic_regression(C=best_c_param_value, tol=1e-10)               # Initialise the Logistic Regression Classifier, for the Best Regularization C Parameter found
    logReg.fit(xs_train[:,:num_features], ys_train)                                 # Fit the Logistic Regression Classifier with the Training Set
    ys_logReg_predict_prob = logReg.predict_proba(xs_test[:,:num_features])[:,1]    # Predict the Probabilities of the Features of the Testing Set, belongs to a certain Class
    logReg_predict_classes_xs_test = logReg.predict(xs_test)                        # Predict and Classify the Values of the Testing Set, with the Logistic Regression Classifier TODO Confirmar
    
    
    # Estimate the Testing Error, based on a certain type of Scoring
    
    # 1) Brier Scoring
    if(score_type == 'brier_score'):
        estimated_true_test_error = skl_brier_score_loss(ys_test, ys_logReg_predict_prob)   # Estimate the Testing Error, related to its Brier Score

    # 2) Logistic Regression Scoring
    if(score_type == 'logistic_regression_score'):
        estimated_accuracy_test = logReg.score(xs_test, ys_test)               # Compute the Training Set's Accuracy (Score), for the Logistic Regression
        estimated_true_test_error = ( 1 - estimated_accuracy_test )            # Compute the Training Error, regarding its Accuracy (Score)
    
    
    num_samples_test_set = len(xs_test)                                        # The Number of Samples, from the Testing Set 
    logReg_num_incorrect_predictions = 0                                       # The Real Number of Incorrect Predictions, regarding the Logistic Regression Classifier
    
    # For each Sample, from the Testing Set
    for current_sample_test in range(num_samples_test_set):
        
        # If the Prediction/Classification of the Class for the current Sample, of the Testing Set is different from the Real Class of the same,
        # it's considered an Real Error in Prediction/Classification, regarding the Logistic Regression Classifier
        if(logReg_predict_classes_xs_test[current_sample_test] != ys_test[current_sample_test] ):
            logReg_num_incorrect_predictions += 1
            
    
    # Return the Predictions of the Samples,
    # the Real Number of Incorrect Predictions and the Estimated True/Test Error, for the Logistic Regression Classifier
    return logReg_predict_classes_xs_test, logReg_num_incorrect_predictions, estimated_true_test_error


# Perform the Classification Process for
# the Logistic Regression Classifier
def do_logistic_regression(xs_test_features_std, ys_train_classes, xs_train_features_std, ys_test_classes, num_features):
       
    k_folds = skl_model_selection.StratifiedKFold(n_splits = NUM_FOLDS)     # The K Folds Combinations Model, for the Stratified K Folds process

    logReg_best_c = 1e10                                    # The Best Regularization Parameter C found, for Logistic Regreession
    logReg_best_valid_error_avg_folds = 1e10                # The Best Average of the Validation Error, for Logistic Regreession
    initial_exp_factor = 0                                  # The Initial Exponential Factor, for the Loop
    final_exp_factor = 15                                   # The Final Exponential Factor, for the Loop
    initial_c_param_value = 1e-2                            # The Initial Regularization Parameter C (i.e., 1e-2)

    logReg_train_error_values = np.zeros((NUM_STEPS_C_REGULARIZATION_LOGISTIC_REGRESSION,2))        # The Values of Training and Validation Errors, for Logistic Regression
    logReg_valid_error_values = np.zeros((NUM_STEPS_C_REGULARIZATION_LOGISTIC_REGRESSION,2))
    
    
    # The loop for try all the Regularization Parameter Cs
    for current_exp_factor in range(initial_exp_factor, final_exp_factor):
    
       
        logReg_train_error_sum = 0                                                        # The sum of the Training and Validation Errors, for Logistic Regression
        logReg_valid_error_sum = 0
        
        current_c_param_value = ( initial_c_param_value * 10**(current_exp_factor) )      # The current Regularization Parameter C

        
        # The loop for all the combinations of K Folds, in the Stratified K Folds process
        for train_idx, valid_idx in k_folds.split(ys_train_classes, ys_train_classes):
            
            # Compute the Training and Validation Errors, for Logistic Regression
            logReg_train_error, logReg_valid_error = compute_logReg_errors(xs_train_features_std, ys_train_classes, train_idx, valid_idx, current_c_param_value, num_features, 'brier_score')
            
            # Sum the current Training and Validation Errors to the Sums of them
            logReg_train_error_sum += logReg_train_error
            logReg_valid_error_sum += logReg_valid_error
            
            
        # Compute the Average of the Sums of the Training and Validation Errors, by the Total Number of Folds 
        logReg_train_error_avg_folds = (logReg_train_error_sum / NUM_FOLDS)
        logReg_valid_error_avg_folds = (logReg_valid_error_sum / NUM_FOLDS)

        
        # Updates the Best Validation Error and also, the Best Regularization C Parameter
        if(logReg_best_valid_error_avg_folds > logReg_valid_error_avg_folds):
            logReg_best_valid_error_avg_folds = logReg_valid_error_avg_folds
            logReg_best_c = current_c_param_value
        
        
        logReg_train_error_values[current_exp_factor, 0] = np.log(current_c_param_value)        # Store the Values for x and y, for all the Training Error values,
        logReg_train_error_values[current_exp_factor, 1] = logReg_train_error_avg_folds         # for the Plot of the Training Errors, as a Function of Logarithm of the C Parameter
        
        logReg_valid_error_values[current_exp_factor, 0] = np.log(current_c_param_value)        # Store the Values for x and y, for all the Validation Error values,
        logReg_valid_error_values[current_exp_factor, 1] = logReg_valid_error_avg_folds         # for the Plot of the Validation Errors, as a Function of Logarithm of the C Parameter
        


    # Plot the Training and Validation Errors, for the Logistic Regression Classifier
    plot_train_valid_error_logistic_regression(logReg_train_error_values, logReg_valid_error_values)

    # Compute the Predictions of the Samples, the Real Number of Incorrect Predictions and the Estimated True/Test Error, of the Testing Set, for the Logistic Regression Classifier
    logReg_predict_classes_xs_test, logReg_num_incorrect_predictions, estimated_logReg_true_test_error = estimate_logReg_true_test_error(xs_train_features_std, ys_train_classes, xs_test_features_std, ys_test_classes, num_features, logReg_best_c, 'brier_score')    
    
    
    # Return the Predictions of the Samples, of the Testing Set, for the Logistic Regression Classifier
    return logReg_train_error_avg_folds, logReg_valid_error_avg_folds, logReg_best_c, logReg_best_valid_error_avg_folds, logReg_predict_classes_xs_test, logReg_num_incorrect_predictions, estimated_logReg_true_test_error
