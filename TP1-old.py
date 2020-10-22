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


# Definition of the necessary Python Libraries

# a) General Libraries:

# Import NumPy Python's Library as np
import numpy as np

# Import Math Python's Library as mathematics
import math as mathematics

# Import PyPlot Sub-Module, from Matplotlib Python's Library as plt
import matplotlib.pyplot as plt

# Import SciKit-Learn as skl
import sklearn as skl

# Import Model Selection Sub-Module, from SciKit-Learn Python's Library,
# as skl_model_selection 
from sklearn import model_selection as skl_model_selection

# Import Brier Score Loss (Metrics) Sub-Module,
# from SciKit-Learn Python's Library, as skl_brier_score_loss
from sklearn.metrics import brier_score_loss as skl_brier_score_loss

# Import Accuracy Score (Metrics) Sub-Module,
# from SciKit-Learn Python's Library, as skl_accuracy_score
from sklearn.metrics import accuracy_score as skl_accuracy_score



# b) Classifiers

# b.1) Logistic Regression Classifier

# Import Logistic Regression Sub-Module, from SciKit-Learn Python's Library,
# as skl_logistic_regression 
from sklearn.linear_model import LogisticRegression as skl_logistic_regression

# b.2) Naïve Bayes Classifier, with customised KDE (Kernel Density Estimation) 

# Import the Kernel Density (Neighbors) Sub-Module,
# from SciKit-Learn Python's Library, as kernel_density
from sklearn.neighbors import KernelDensity as skl_kernel_density

# b.3) Gaussian Naïve Bayes Classifier

# Import GaussianNB (Naïve Bayes) Sub-Module,
# from SciKit-Learn Python's Library, as skl_gaussian_naive_bayes 
from sklearn.naive_bayes import GaussianNB as skl_gaussian_naive_bayes


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Constants #1

# The Number of Folds, for Stratified K Folds, in Cross-Validation
NUM_FOLDS = 5

# The Number of Steps/Variations for ajusting the C Regularization parameter,
# for the Logistic Regression
NUM_STEPS_C_REGULARIZATION_LOGISTIC_REGRESSION = 15

# The Number of Steps/Variations for ajusting the Bandwidth parameter,
# for the Naïve Bayes
NUM_STEPS_BANDWIDTH_NAIVE_BAYES = 30

# The Boolean Flag for Debugging
DEBUG_FLAG = True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# The files of the Datasets for Training and Testing

# The Data for Training Set
train_set_data_file = "files/data/TP1_train.tsv"

# The Data for Testing Set
test_set_data_file = "files/data/TP1_test.tsv"



# Load the Data for Training Set with NumPy function loadtxt
train_set_data_not_random = np.loadtxt(train_set_data_file, delimiter="\t")

# Load the Data for Testing Set with NumPy function loadtxt
test_set_data_not_random = np.loadtxt(test_set_data_file, delimiter="\t")


# If the Boolean Flag for Debugging is set to True,
# print some relevant information
if(DEBUG_FLAG == True):

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



# If the Boolean Flag for Debugging is set to True,
# print some relevant information
if(DEBUG_FLAG == True):

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


# Select the Classes of the Training Set, randomized
ys_train_classes = train_set_data_random[:,-1]

# Select the Features of the Training Set, randomized
xs_train_features = train_set_data_random[:,0:-1]


# If the Boolean Flag for Debugging is set to True,
# print some relevant information
if(DEBUG_FLAG == True):

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


# Select the Classes of the Testing Set, randomized
ys_test_classes = test_set_data_random[:,-1]

# Select the Features of the Testing Set, randomized
xs_test_features = test_set_data_random[:,0:-1]

# The size of the Data for Testing Set, randomized
test_set_size = len(xs_test_features)


# If the Boolean Flag for Debugging is set to True,
# print some relevant information
if(DEBUG_FLAG == True):

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


# If the Boolean Flag for Debugging is set to True,
# print some relevant information
if(DEBUG_FLAG == True):

    # Print the Standardized Features of the Data for Training Set, randomized
    print("\n")
    print("The Standardized Features of the Data for Training Set, randomized:")
    print(xs_train_features_std)
    
    # Print a new/blank line
    print ("\n")


# Standardize the Testing Set, randomized
xs_test_features_std = ( ( xs_test_features - train_means ) / train_stdevs )


# If the Boolean Flag for Debugging is set to True,
# print some relevant information
if(DEBUG_FLAG == True):

    # Print the Standardized Features of the Data for Testing Set, randomized
    print("\n")
    print("The Standardized Features of the Data for Testing Set, randomized:")
    print(xs_test_features_std)
    
    # Print a new/blank line
    print ("\n")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Constants #2

# The Number of Features
# (i.e., 4 Features, per each Banknote)
NUM_FEATURES = xs_train_features_std.shape[1]

# The Number of Classes
# (i.e., 2 Classes possible, per each Banknote, Real or Fake)
NUM_CLASSES = len(set(ys_train_classes))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -----------------------------------------------------
# \                                                   \
# \  Classifier 1) - Logistic Regression,             \
# \  varying its Regularization C parameter           \
# \___________________________________________________\

# The Function to Compute and Return the Errors for Training and Validation Sets,
# for the Logistic Regression Classifier
def compute_logistic_regression_errors(xs, ys, train_idx, valid_idx, c_param_value, score_type = 'brier_score'):
    
    # Initialise the Logistic Regression,
    # from the Linear Model of the SciKit-Learn
    logistic_regression = skl_logistic_regression(C = c_param_value, tol = 1e-10)
    
    # Fit the Logistic Regression 
    logistic_regression.fit(xs[train_idx,:NUM_FEATURES], ys[train_idx])
    
    # Compute the prediction probabilities of some Features,
    # belonging to a certain Class, due to the 
    ys_logistic_regression_prediction_probabilities = logistic_regression.predict_proba(xs[:,:NUM_FEATURES])[:,1]
    
    
    # Compute the Training and Validation Errors, based on a certain type of Scoring:
    # 1) Based on Brier Score
    if(score_type == 'brier_score'):
    
        # Compute the Training Error, related to its Brier Score
        logistic_regression_train_error = skl_brier_score_loss(ys[train_idx], ys_logistic_regression_prediction_probabilities[train_idx])

        # Compute the Validation Error, related to its Brier Score        
        logistic_regression_valid_error = skl_brier_score_loss(ys[valid_idx], ys_logistic_regression_prediction_probabilities[valid_idx])

    # 2) Based on Logistic Regression Score
    if(score_type == 'logistic_regression_score'):
        
        # Compute the Training Set's Accuracy (Score), for the Logistic Regression
        logistic_regression_accuracy_train = logistic_regression.score(xs[train_idx], ys[train_idx])
    
        # Compute the Validation Set's Accuracy (Score), for the Logistic Regression    
        logistic_regression_accuracy_valid = logistic_regression.score(xs[valid_idx], ys[valid_idx])
        
        # Compute the Training Error, regarding its Accuracy (Score)
        logistic_regression_train_error = ( 1 - logistic_regression_accuracy_train )
        
        # Compute the Validation Error, regarding its Accuracy (Score)
        logistic_regression_valid_error = ( 1 - logistic_regression_accuracy_valid )

        
    # Return the Training and Validation Errors, for the Logistic Regression
    return logistic_regression_train_error, logistic_regression_valid_error


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
def estimate_logistic_regression_true_test_error(xs_train, ys_train, xs_test, ys_test, best_c_param_value=1e12, score_type = 'brier_score'):
    
    # Initialise the Logistic Regression Classifier,
    # for the Best Regularization C Parameter found
    logistic_regression = skl_logistic_regression(C=best_c_param_value, tol=1e-10)
   
    # Fit the Logistic Regression Classifier with the Training Set
    logistic_regression.fit(xs_train[:,:NUM_FEATURES], ys_train)
    
    # Predict the Probabilities of the Features of the Testing Set, belongs to a certain Class
    ys_logistic_regression_prediction_probabilities = logistic_regression.predict_proba(xs_test[:,:NUM_FEATURES])[:,1]
    
    # Predict and Classify the Values of the Testing Set,
    # with the Logistic Regression Classifier TODO Confirmar
    logistic_regression_prediction_classes_for_samples_xs_test = logistic_regression.predict(xs_test)
    
    
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
    
    
    # The Number of Samples, from the Testing Set 
    num_samples_test_set = len(xs_test)

    # The Real Number of Incorrect Predictions, regarding the Logistic Regression Classifier
    real_logistic_regression_num_incorrect_predictions = 0
    
    # For each Sample, from the Testing Set
    for current_sample_test in range(num_samples_test_set):
        
        # If the Prediction/Classification of the Class for the current Sample,
        # of the Testing Set is different from the Real Class of the same,
        # it's considered an Real Error in Prediction/Classification,
        # regarding the Logistic Regression Classifier
        if(logistic_regression_prediction_classes_for_samples_xs_test[current_sample_test] != ys_test[current_sample_test] ):
            real_logistic_regression_num_incorrect_predictions += 1
            
    
    # Return the Predictions of the Samples,
    # the Real Number of Incorrect Predictions and the Estimated True/Test Error,
    # for the Logistic Regression Classifier
    return logistic_regression_prediction_classes_for_samples_xs_test, real_logistic_regression_num_incorrect_predictions, estimated_true_test_error


# Perform the Classification Process for
# the Logistic Regression Classifier
def do_logistic_regression():
    
    print("-----------------------------------------------------------------")
    print("1) Starting the Logistic Regression Classifier...")
    print("-----------------------------------------------------------------")
    print("\n\n")    
    
    # The K Folds Combinations Model, for the Stratified K Folds process
    k_folds = skl_model_selection.StratifiedKFold(n_splits = NUM_FOLDS)
    
    # The Best Regularization Parameter C found,
    # for Logistic Regreession
    logistic_regression_best_c_param_value = 1e10
    
    # The Best Average of the Validation Error,
    # for Logistic Regreession
    logistic_regression_best_valid_error_avg_folds = 1e10
    
    
    # The Initial Exponential Factor, for the Loop
    initial_exp_factor = 0
    
    # The Final Exponential Factor, for the Loop
    final_exp_factor = 15
    
    # The Initial Regularization Parameter C (i.e., 1e-2)
    initial_c_param_value = 1e-2
    
    # The Values of Training and Validation Errors, for Logistic Regression
    logistic_regression_train_error_values = np.zeros((NUM_STEPS_C_REGULARIZATION_LOGISTIC_REGRESSION,2))
    logistic_regression_valid_error_values = np.zeros((NUM_STEPS_C_REGULARIZATION_LOGISTIC_REGRESSION,2))
    
    
    # The loop for try all the Regularization Parameter Cs
    for current_exp_factor in range(initial_exp_factor, final_exp_factor):
    
        # The sum of the Training and Validation Errors, for Logistic Regression
        logistic_regression_train_error_sum = 0
        logistic_regression_valid_error_sum = 0
        
        # The current Regularization Parameter C
        current_c_param_value = ( initial_c_param_value * 10**(current_exp_factor) )


        # If the Boolean Flag for Debugging is set to True,
        # print some relevant information
        if(DEBUG_FLAG == True):
        
            # Print the information about
            # trying a new Regularization Parameter C, for Logistic Regression
            print("Trying the Regularization Parameter C = {},\nfor Logistic Regression...".format(current_c_param_value))
            print("\n")

        
        # The loop for all the combinations of K Folds, in the Stratified K Folds process
        for train_idx, valid_idx in k_folds.split(ys_train_classes, ys_train_classes):
            
            # Compute the Training and Validation Errors, for Logistic Regression
            logistic_regression_train_error, logistic_regression_valid_error = compute_logistic_regression_errors(xs_train_features_std, ys_train_classes, train_idx, valid_idx, current_c_param_value, 'brier_score')
            
            # Sum the current Training and Validation Errors to the Sums of them
            logistic_regression_train_error_sum += logistic_regression_train_error
            logistic_regression_valid_error_sum += logistic_regression_valid_error
            
            
        # Compute the Average of the Sums of the Training and Validation Errors, by the Total Number of Folds 
        logistic_regression_train_error_avg_folds = ( logistic_regression_train_error_sum / NUM_FOLDS )
        logistic_regression_valid_error_avg_folds = ( logistic_regression_valid_error_sum / NUM_FOLDS )
        

        # If the Boolean Flag for Debugging is set to True,
        # print some relevant information
        if(DEBUG_FLAG == True):
            
            # Print the information about
            # the Current Value for Regularization Parameter C, for Logistic Regression
            print("Current Value for Regularization C = {} :".format(current_c_param_value))
            print("- Training Error = {} ; - Validation Error = {}".format(logistic_regression_train_error_avg_folds, logistic_regression_valid_error_avg_folds))
            print("\n")

        
        # Updates the Best Validation Error and also, the Best Regularization C Parameter
        if(logistic_regression_best_valid_error_avg_folds > logistic_regression_valid_error_avg_folds):
            logistic_regression_best_valid_error_avg_folds = logistic_regression_valid_error_avg_folds
            logistic_regression_best_c_param_value = current_c_param_value
            

        # If the Boolean Flag for Debugging is set to True,
        # print some relevant information
        if(DEBUG_FLAG == True):

            # Print the information about
            # Storing the Training and Validation Errors, for the future Plot of Training and Validation Errors
            print("Storing the Training and Validation Errors, for the future Plot of Training and Validation Errors...")
            print("\n")
        
        
        # Store the Values for x and y, for all the Training Error values,
        # for the Plot of the Training Errors, as a Function of Logarithm of the C Parameter
        logistic_regression_train_error_values[current_exp_factor, 0] = np.log(current_c_param_value)
        logistic_regression_train_error_values[current_exp_factor, 1] = logistic_regression_train_error_avg_folds

        # Store the Values for x and y, for all the Validation Error values,
        # for the Plot of the Validation Errors, as a Function of Logarithm of the C Parameter
        logistic_regression_valid_error_values[current_exp_factor, 0] = np.log(current_c_param_value)
        logistic_regression_valid_error_values[current_exp_factor, 1] = logistic_regression_valid_error_avg_folds
        
    
    # If the Boolean Flag for Debugging is set to True,
    # print some relevant information
    if(DEBUG_FLAG == True):            

        # Print the Best Value for the Regularization C Parameter
        print("\n")
        print("Best Value for Regularization C = {} :".format(logistic_regression_best_c_param_value))
        print("- Best Validation Error = {}".format(logistic_regression_best_valid_error_avg_folds))
        print("\n")


    # Plot the Training and Validation Errors, for the Logistic Regression Classifier
    plot_train_valid_error_logistic_regression(logistic_regression_train_error_values, logistic_regression_valid_error_values)

    # Compute the Predictions of the Samples,
    # the Real Number of Incorrect Predictions and the Estimated True/Test Error,
    # of the Testing Set, for the Logistic Regression Classifier
    logistic_regression_prediction_classes_for_samples_xs_test, real_logistic_regression_num_incorrect_predictions, estimated_logistic_regression_true_test_error = estimate_logistic_regression_true_test_error(xs_train_features_std, ys_train_classes, xs_test_features_std, ys_test_classes, logistic_regression_best_c_param_value, 'brier_score')    

    # If the Boolean Flag for Debugging is set to True,
    # print some relevant information
    if(DEBUG_FLAG == True):

        # Print the Estimated True/Test Error
        print("\n")
        print("- Estimated True/Test Error = {}".format(estimated_logistic_regression_true_test_error))
        print("- Number of Incorrect Predictions (Number of real Errors) = {}".format(real_logistic_regression_num_incorrect_predictions))
    
    
    # The number of the Samples, from the Testing Set
    num_samples_test_set = len(xs_test_features_std)  

    # Computes the Aproximate Normal Test, for the Logistic Regression Classifier
    logistic_regression_aproximate_normal_test_deviation_lower_bound, logistic_regression_aproximate_normal_test_deviation_upper_bound = aproximate_normal_test(real_logistic_regression_num_incorrect_predictions, estimated_logistic_regression_true_test_error, num_samples_test_set)
    
    # If the Boolean Flag for Debugging is set to True,
    # print some relevant information
    if(DEBUG_FLAG == True):   
        # Print the Approximate Normal Test, with Confidence Level of 95% and
        # its Interval range of values, for the Test itself
        print("\n")
        print("- Approximate Normal Test, with Confidence Level of 95% = [ {} - {} ; {} + {} ]".format(real_logistic_regression_num_incorrect_predictions, logistic_regression_aproximate_normal_test_deviation_upper_bound, real_logistic_regression_num_incorrect_predictions, logistic_regression_aproximate_normal_test_deviation_upper_bound))
        print("- Approximate Normal Test Interval = [ {} ; {} ]".format( ( real_logistic_regression_num_incorrect_predictions + logistic_regression_aproximate_normal_test_deviation_lower_bound ) , ( real_logistic_regression_num_incorrect_predictions + logistic_regression_aproximate_normal_test_deviation_upper_bound ) ))
    
    
    # Return the Predictions of the Samples,
    # of the Testing Set, for the Logistic Regression Classifier
    return logistic_regression_prediction_classes_for_samples_xs_test


# -------------------------------------------------------
# \                                                     \
# \  Classifier 2) - Naïve Bayes,                       \
# \  with customised KDEs (Kernel Density Estimations), \
# \  varying its Bandwidth Regularization parameter     \
# \_____________________________________________________\

# The Function to compute the 
def compute_naive_bayes_errors(xs, ys, train_idx, valid_idx, bandwidth_param_value):
    
    # Initialise the List of Logarithms of Base e of Prior Probabilities of
    # the Occurrence for each Class, in the Training Set,
    # for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    naive_bayes_logs_prior_probabilities_classes_occurrences_train_list = []
    
    
    # Initialise the Kernel Density Estimation (KDE),
    # with current Bandwidth Regularization Parameter
    kernel_density_estimation = skl_kernel_density(bandwidth=bandwidth_param_value, kernel='gaussian')
    
    
    # In order to compute the Errors of the Naïve Bayes,
    # it's needed to work with each pair of (Class, Feature) 
    
    # As, we have 2 classes and 4 features,
    # we will need a total of 8 Kernel Density Estimations (KDEs)
    # (2 Classes x 4 Features) = 8 Kernel Density Estimations)
    
    
    # The Number of Samples of the whole Training Set
    num_samples_all_xs_train = len(xs)


    # The Features of the Training Set
    xs_train = xs[train_idx]
    
    # The Classes of the Training Set
    ys_train = ys[train_idx]
    
    # The Number of Samples of the Training Set
    num_samples_xs_train = len(xs_train)
    
    
    # The Classes of the Validation Set
    ys_valid = ys[valid_idx]
    
    
    # The Logarithm Densities per each Class,
    # for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    naive_bayes_logarithm_densities_per_class = np.zeros((num_samples_all_xs_train, NUM_CLASSES))                                
    
    # The Classifications/Predictions of the Samples of the whole Training Set,
    # for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    naive_bayes_predictions_all_xs_train_samples = np.zeros((num_samples_all_xs_train))
    
    
    # In order to compute the Errors of the Naïve Bayes,
    # it's needed to work with each pair of (Class, Feature) 
    
    # As, we have 2 classes and 4 features,
    # we will need a total of 8 Kernel Density Estimations (KDEs)
    # (2 Classes x 4 Features) = 8 Kernel Density Estimations)
    
    # For each possible Class of the Dataset
    for current_class in range(NUM_CLASSES):
    
        # The Samples of the Training Set, for the Current Class
        xs_train_current_class = xs_train[ys_train == current_class]
        
        # Compute the Probabilities of the Occurrence for each Class,
        # in the whole Training Set, for the Naïve Bayes Classifier,
        # with custom KDEs (Kernel Density Estimations)
        naive_bayes_prior_probability_occurrences_for_current_class_train = ( len(xs_train[ys_train == current_class]) / num_samples_xs_train )
                
        # Compute the Logarithm of Base e of Prior Probabilities of
        # the Occurrence for each Class, in the whole Training Set, to the respectively List for each Class,
        # and append it to the respective List
        naive_bayes_logs_prior_probabilities_classes_occurrences_train_list.append( np.log(naive_bayes_prior_probability_occurrences_for_current_class_train) )
        
        
        # For each possible Feature of the Dataset
        for current_feature in range(NUM_FEATURES):
            
            # Fit the current Kernel Density Estimation (KDE), with the whole Training Set,
            # for the current pair (Class, Feature), for the Naïve Bayes Classifier,
            # with custom KDEs (Kernel Density Estimations)
            kernel_density_estimation.fit(xs_train_current_class[:,[current_feature]])
            
            # Compute and sum the Logarithm Densities for the current pair (Class, Feature),
            # for the current KDE (Kernel Density Estimation), for the whole Training Set
            naive_bayes_logarithm_densities_per_class[:, current_class] += kernel_density_estimation.score_samples(xs[:, [current_feature]])
        
        
        # Sum the Logarithm of Base e of Prior Probabilities of
        # the Occurrence for each Class, in the whole Training Set,
        # to the Logarithm Densities per each Class,
        # for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
        naive_bayes_logarithm_densities_per_class[:, current_class] += naive_bayes_logs_prior_probabilities_classes_occurrences_train_list[current_class]

    
    # For each Sample of the whole Training Set, try to predict its Class
    for current_sample_x_all_xs_train in range(num_samples_all_xs_train):
        
        # Predict the current Sample of the whole Training Set, as the Maximum Argument (i.e., the Class) of it,
        # i.e. the argument/index with the highest probability of the Predictions of the Classes for each Sample
        naive_bayes_predictions_all_xs_train_samples[current_sample_x_all_xs_train] = np.argmax( naive_bayes_logarithm_densities_per_class[current_sample_x_all_xs_train] )
    
    
    # The Classifications/Predictions of the Samples of the Training Set,
    # for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    naive_bayes_predictions_xs_train_samples = naive_bayes_predictions_all_xs_train_samples[train_idx]
    
    # Compute the Accuracy of Score for the Predictions of the Classes for the Training Set  
    naive_bayes_accuracy_train = skl_accuracy_score(ys_train, naive_bayes_predictions_xs_train_samples)
    
    # Compute the Training Error, regarding the Accuracy Score for
    # the Predictions of the Classes for the Training Set  
    naive_bayes_error_train = ( 1 - naive_bayes_accuracy_train )


    # The Classifications/Predictions of the Samples of the Validation Set,
    # for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    naive_bayes_predictions_xs_valid_samples = naive_bayes_predictions_all_xs_train_samples[valid_idx]
       
    # Compute the Accuracy of Score for the Predictions of the Classes for the Validation Set  
    naive_bayes_accuracy_valid = skl_accuracy_score(ys_valid, naive_bayes_predictions_xs_valid_samples)
    
    # Compute the Validation Error, regarding the Accuracy Score for
    # the Predictions of the Classes for the Validation Set  
    naive_bayes_error_valid = ( 1 - naive_bayes_accuracy_valid )
    
    
    # Return the Training and Validation Errors for the Naïve Bayes
    return naive_bayes_error_train, naive_bayes_error_valid
    

# The Function to Plot the Training and Validation, for the Naïve Bayes
def plot_train_valid_error_naive_bayes(train_error_values, valid_error_values):
    
    # Initialise the Plot
    plt.figure(figsize=(8, 8), frameon=True)

    # Set the line representing the continuous values,
    # for the Functions of the Training and Validation Errors
    plt.plot(train_error_values[:,0], train_error_values[:,1],'-', color="blue")
    plt.plot(valid_error_values[:,0], valid_error_values[:,1],'-', color="red")
    
    # Set the axis for the Plot
    plt.axis([min(valid_error_values[:,0]), max(valid_error_values[:,0]), min(valid_error_values[:,1]), max(valid_error_values[:,1])])
    
    # Set the laber for the X axis of the Plot
    plt.xlabel("Bandwidth")
    
    # Set the laber for the Y axis of the Plot
    plt.ylabel("Training/Validation Errors")
    
    # Set the Title of the Plot
    plt.title('Naïve Bayes, with custom Kernel Density Estimations, varying the Bandwidth parameter\n\nTraining Error (Blue) / Cross-Validation Error (Red)')
    
    # Save the Plot, as a figure/image
    plt.savefig('files/imgs/NB.png', dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()


# The Function to Estimate the True/Test Error of the Testing Set,
# for the Naïve Bayes Classifier
def estimate_naive_bayes_true_test_error(xs_train, ys_train, xs_test, ys_test, best_bandwidth_param_value):
    
    # Initialise the List of Logarithms of Base e of Prior Probabilities of
    # the Occurrence for each Class, in the Testing Set
    naive_bayes_logs_prior_probabilities_classes_occurrences_test_list = []
    
    # Initialise the Kernel Density Estimation (KDE),
    # with current Bandwidth Regularization Parameter
    kernel_density_estimation = skl_kernel_density(bandwidth=best_bandwidth_param_value, kernel='gaussian')
        
    # TODO len(train) ou len(test) ???    
    # The Number of Samples, in the Testing Set
    num_samples_xs_test = len(xs_test)
  
    # The Logarithm Densities per each Class, in the Testing Set
    naive_bayes_logarithm_densities_per_class_test = np.zeros((num_samples_xs_test, NUM_CLASSES))                                
        
        
    # The Classification/Prediction of the Samples, in the Testing Set,
    # to the respective Classes
    naive_bayes_prediction_classes_for_samples_xs_test = np.zeros((num_samples_xs_test))
    

    # In order to compute the Errors of the Naïve Bayes,
    # it's needed to work with each pair of (Class, Feature) 
    
    # As, we have 2 classes and 4 features,
    # we will need a total of 8 Kernel Density Estimations (KDEs)
    # (2 Classes x 4 Features) = 8 Kernel Density Estimations)
    
    # For each possible Class of the Dataset
    for current_class in range(NUM_CLASSES):
        
        xs_train_current_class = xs_train[ys_train == current_class]
        
        # Compute the Probabilities of the Occurrence for each Class,
        # in the Testing Set
        naive_bayes_prior_probability_occurrences_for_current_class_train = ( len(xs_train[ys_train == current_class]) / num_samples_xs_test )
                
        # Compute the Logarithm of Base e of Prior Probabilities of
        # the Occurrence for each Class, in the Testing Set, to the respectively List for each Class,
        # and append it to the respective List
        naive_bayes_logs_prior_probabilities_classes_occurrences_test_list.append( np.log(naive_bayes_prior_probability_occurrences_for_current_class_train) )
        
        
        # For each possible Feature of the Dataset
        for current_feature in range(NUM_FEATURES):
            
            # Fit the Kernel Density Estimation (KDE), with the Testing Set
            kernel_density_estimation.fit(xs_train_current_class[:, current_feature].reshape(-1,1))
          
            # Compute and sum the Logarithm Densities for the current pair (Class, Feature),
            # for the current KDE (Kernel Density Estimation), for the Testing Set
            naive_bayes_logarithm_densities_per_class_test[:, current_class] += kernel_density_estimation.score_samples(xs_test[:, [current_feature]])
        
        
        # Sum the Logarithm of Base e of Prior Probabilities of
        # the Occurrence for each Class, in the Testing Set,
        # to the Logarithm Densities per each Class,
        # for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
        naive_bayes_logarithm_densities_per_class_test[:, current_class] += naive_bayes_logs_prior_probabilities_classes_occurrences_test_list[current_class]

    
    # For each Sample of the Testing Set, try to predict its Class
    for current_sample_x_test in range(num_samples_xs_test):
        
        # Predict the current Sample of the Testing Set, as the Maximum Argument (i.e., the Class) of it,
        # i.e. the argument/index with the highest probability of the Predictions of the Classes for each Sample,
        # in the Testing Set
        naive_bayes_prediction_classes_for_samples_xs_test[current_sample_x_test] = np.argmax( naive_bayes_logarithm_densities_per_class_test[current_sample_x_test] )
    
    
    # Compute the Accuracy of Score for the Predictions of the Classes for the Testing Set  
    naive_bayes_estimated_accuracy_test = skl_accuracy_score(ys_test, naive_bayes_prediction_classes_for_samples_xs_test)
    
    # Compute the Estimated True Testing Error, regarding the Accuracy Score for
    # the Predictions of the Classes for the Testing Set  
    naive_bayes_estimated_true_error_test = ( 1 - naive_bayes_estimated_accuracy_test )
    
    
    # The Number of Samples, from the Testing Set 
    num_samples_test_set = len(xs_test)

    # The Real Number of Incorrect Predictions,
    # regarding the Naïve Bayes Classifier,
    # with custom KDEs (Kernel Density Estimations)
    real_naive_bayes_num_incorrect_predictions = 0
    
    # For each Sample, from the Testing Set
    for current_sample_test in range(num_samples_test_set):
        
        # If the Prediction/Classification of the Class for the current Sample,
        # of the Testing Set is different from the Real Class of the same,
        # it's considered an Real Error in Prediction/Classification,
        # regarding the Naïve Bayes Classifier,
        # with custom KDEs (Kernel Density Estimations)
        if(naive_bayes_prediction_classes_for_samples_xs_test[current_sample_test] != ys_test[current_sample_test] ):
            real_naive_bayes_num_incorrect_predictions += 1
    
    
    # Return the Predictions of the Samples,
    # the Real Number of Incorrect Predictions and the Estimated True/Test Error,
    # for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    return naive_bayes_prediction_classes_for_samples_xs_test, real_naive_bayes_num_incorrect_predictions, naive_bayes_estimated_true_error_test


def do_naive_bayes():
    
    print("-----------------------------------------------------------------")
    print("2) Starting the Naïve Bayes Classifier...")
    print("-----------------------------------------------------------------")
    print("\n\n")
    
    
    # The K Folds Combinations Model, for the Stratified K Folds process
    k_folds = skl_model_selection.StratifiedKFold(n_splits = NUM_FOLDS)
    
    
    # The Best Regularization Parameter Bandwidth found,
    # for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    naive_bayes_best_bandwidth_param_value = 1e10
    
    # The Best Average of the Validation Error,
    # for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    naive_bayes_best_valid_error_avg_folds = 1e10
    
    
    # The Values of Training and Validation Errors,
    # for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    naive_bayes_train_error_values = np.zeros((NUM_STEPS_BANDWIDTH_NAIVE_BAYES, 2))
    naive_bayes_valid_error_values = np.zeros((NUM_STEPS_BANDWIDTH_NAIVE_BAYES, 2))
    
    
    # The initial factor of each Bandwidth Step,
    # for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    initial_bandwidth = 2e-2
    
    # The final factor of each Bandwidth Step,
    # for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    final_bandwidth = 6e-1
    
    # The factor of each Bandwidth Step,
    # for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    bandwidth_step = 2e-2

    
    # The Number of the current Bandwidth Step,
    # for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    current_step_bandwidth_naive_bayes = 0


    # The loop for try all the Regularization Parameter Bandwidths
    for current_bandwidth_param_value in np.arange(initial_bandwidth, ( final_bandwidth + bandwidth_step ), bandwidth_step):
        
        # The sum of the Training and Validation Errors,
        # for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
        naive_bayes_train_error_sum = 0
        naive_bayes_valid_error_sum = 0
        
        
        # If the Boolean Flag for Debugging is set to True,
        # print some relevant information
        if(DEBUG_FLAG == True):
        
            # Print the information about
            # trying a new Regularization Parameter Bandwidth,
            # for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
            print("Trying the Regularization Parameter Bandwidth = {},\nfor Naïve Bayes...".format(current_bandwidth_param_value))
            print("\n")
        
        # The loop for all the combinations of K Folds, in the Stratified K Folds process
        for train_idx, valid_idx in k_folds.split(ys_train_classes, ys_train_classes):
            
            # Compute the Training and Validation Errors, for Naïve Bayes,
            # with custom KDEs (Kernel Density Estimations)
            naive_bayes_train_error, naive_bayes_valid_error = compute_naive_bayes_errors(xs_train_features_std, ys_train_classes, train_idx, valid_idx, current_bandwidth_param_value)
            
            # Sum the current Training and Validation Errors to the Sums of them
            naive_bayes_train_error_sum += naive_bayes_train_error
            naive_bayes_valid_error_sum += naive_bayes_valid_error
            
            
        # Compute the Average of the Sums of the Training and Validation Errors, by the Total Number of Folds
        naive_bayes_train_error_avg_folds = ( naive_bayes_train_error_sum / NUM_FOLDS )
        naive_bayes_valid_error_avg_folds = ( naive_bayes_valid_error_sum / NUM_FOLDS )

        
        # If the Boolean Flag for Debugging is set to True,
        # print some relevant information
        if(DEBUG_FLAG == True):
            
            # Print the information about
            # the Current Value for Regularization Parameter Bandwidth, for Naïve Bayes, with Kernel Density Estimations
            print("Current Value for Regularization Bandwidth = {} :".format(current_bandwidth_param_value))
            print("- Training Error = {} ; - Validation Error = {}".format(naive_bayes_train_error_avg_folds, naive_bayes_valid_error_avg_folds))
            print("\n")
            
        # Updates the Best Validation Error and also, the Best Regularization Bandwidth Parameter
        if(naive_bayes_best_valid_error_avg_folds > naive_bayes_valid_error_avg_folds):
            naive_bayes_best_valid_error_avg_folds = naive_bayes_valid_error_avg_folds
            naive_bayes_best_bandwidth_param_value = current_bandwidth_param_value
            

        # If the Boolean Flag for Debugging is set to True,
        # print some relevant information
        if(DEBUG_FLAG == True):

            # Print the information about
            # Storing the Training and Validation Errors, for the future Plot of Training and Validation Errors
            print("Storing the Training and Validation Errors, for the future Plot of Training and Validation Errors...")
            print("\n")
        
        
        # Store the Values for x and y, for all the Training Error values,
        # for the Plot of the Training Errors, as a Function of Logarithm of the Bandwidth Parameter
        naive_bayes_train_error_values[current_step_bandwidth_naive_bayes, 0] = current_bandwidth_param_value
        naive_bayes_train_error_values[current_step_bandwidth_naive_bayes, 1] = naive_bayes_train_error_avg_folds

        # Store the Values for x and y, for all the Validation Error values,
        # for the Plot of the Validation Errors, as a Function of Logarithm of the Bandwidth Parameter
        naive_bayes_valid_error_values[current_step_bandwidth_naive_bayes, 0] = current_bandwidth_param_value
        naive_bayes_valid_error_values[current_step_bandwidth_naive_bayes, 1] = naive_bayes_valid_error_avg_folds
        
        # Increment the Current Step of the Bandwidth Parameter value
        current_step_bandwidth_naive_bayes += 1
    
    # If the Boolean Flag for Debugging is set to True,
    # print some relevant information
    if(DEBUG_FLAG == True):            

        # Print the Best Value for the Regularization Bandwidth Parameter
        print("\n")
        print("Best Value for Regularization Bandwidth = {} :".format(naive_bayes_best_bandwidth_param_value))
        print("- Best Validation Error = {}".format(naive_bayes_best_valid_error_avg_folds))
        print("\n")
        
        
    # Plot the Training and Validation Errors, for the Naïve Bayes Classifier
    plot_train_valid_error_naive_bayes(naive_bayes_train_error_values, naive_bayes_valid_error_values)
    
    # Compute the Predictions of the Samples,
    # the Real Number of Incorrect Predictions and the Estimated True/Test Error,
    # for the Testing Set, of the Naïve Bayes Regression Classifier,
    # with custom KDEs (Kernel Density Estimations)
    naive_bayes_prediction_classes_for_samples_xs_test, real_naive_bayes_num_incorrect_predictions, estimated_naive_bayes_true_test_error = estimate_naive_bayes_true_test_error(xs_train_features_std, ys_train_classes, xs_test_features_std, ys_test_classes, naive_bayes_best_bandwidth_param_value)    
    

    # If the Boolean Flag for Debugging is set to True,
    # print some relevant information
    if(DEBUG_FLAG == True):

        # Print the Estimated True/Test Error
        print("\n")
        print("- Estimated True/Test Error = {}".format(estimated_naive_bayes_true_test_error))
        print("- Number of Incorrect Predictions (Number of real Errors) = {}".format(real_naive_bayes_num_incorrect_predictions))
    
    
    # The number of the Samples, from the Testing Set
    num_samples_test_set = len(xs_test_features_std)  

    # Computes the Aproximate Normal Test,
    # for the Naïve Bayes Classifier,
    # with custom KDEs (Kernel Density Estimations)
    naive_bayes_aproximate_normal_test_deviation_lower_bound, naive_bayes_aproximate_normal_test_deviation_upper_bound = aproximate_normal_test(real_naive_bayes_num_incorrect_predictions, estimated_naive_bayes_true_test_error, num_samples_test_set)
    
    # If the Boolean Flag for Debugging is set to True,
    # print some relevant information
    if(DEBUG_FLAG == True):   
        # Print the Approximate Normal Test, with Confidence Level of 95% and
        # its Interval range of values, for the Test itself
        print("\n")
        print("- Approximate Normal Test, with Confidence Level of 95% = [ {} - {} ; {} + {} ]".format(real_naive_bayes_num_incorrect_predictions, naive_bayes_aproximate_normal_test_deviation_upper_bound, real_naive_bayes_num_incorrect_predictions, naive_bayes_aproximate_normal_test_deviation_upper_bound))
        print("- Approximate Normal Test Interval = [ {} ; {} ]".format( ( real_naive_bayes_num_incorrect_predictions + naive_bayes_aproximate_normal_test_deviation_lower_bound ) , ( real_naive_bayes_num_incorrect_predictions + naive_bayes_aproximate_normal_test_deviation_upper_bound ) ))
    
    
    # Return the Predictions of the Samples,
    # of the Testing Set, for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    return naive_bayes_prediction_classes_for_samples_xs_test

    

# -----------------------------------------------------
# \                                                   \
# \  Classifier 3) - Gaussian Naïve Bayes             \
# \___________________________________________________\


# The Function to Compute and Return the Errors for Training and Validation Sets,
# for the Gaussian Naïve Bayes Classifier
def compute_gaussian_naive_bayes_errors(gaussian_naive_bayes, xs, ys, train_idx, valid_idx):
    
    # Fit the Gaussian Naïve Bayes, with the Training Set
    gaussian_naive_bayes.fit(xs[train_idx], ys[train_idx])
    
    # Compute the Training Set's Accuracy (Score), for the Gaussian Naïve Bayes    
    gaussian_naive_bayes_train_accuracy = gaussian_naive_bayes.score(xs[train_idx], ys[train_idx])
    
    # Compute the Training Error, regarding its Accuracy (Score), for the Gaussian Naïve Bayes 
    gaussian_naive_bayes_train_error = ( 1 - gaussian_naive_bayes_train_accuracy )


    # Compute the Validation Set's Accuracy (Score),
    # for the Gaussian Naïve Bayes    
    gaussian_naive_bayes_valid_accuracy = gaussian_naive_bayes.score(xs[valid_idx], ys[valid_idx])
    
    # Compute the Validation Error, regarding its Accuracy (Score),
    # for the Gaussian Naïve Bayes 
    gaussian_naive_bayes_valid_error = ( 1 - gaussian_naive_bayes_valid_accuracy )

    
    # Return the Training and Validation Errors, for the Gaussian Naïve Bayes
    return gaussian_naive_bayes_train_error, gaussian_naive_bayes_valid_error


# The Function to Estimate the True/Test Error of the Testing Set,
# for the Gaussian Naïve Bayes Classifier
def estimate_gaussian_naive_bayes_true_test_error(xs_train, ys_train, xs_test, ys_test):
    
    # Initialise the Gaussian Naïve Bayes Classifier
    gaussian_naive_bayes = skl_gaussian_naive_bayes()
    
    # Fit the Gaussian Naïve Bayes, with the Testing Set
    gaussian_naive_bayes.fit(xs_train, ys_train)
    
    # Predict and Classify the Values of the Testing Set,
    # with the Gaussian Naïve Bayes Classifier TODO Confirmar
    gaussian_naive_bayes_prediction_classes_for_samples_xs_test = gaussian_naive_bayes.predict(xs_test)
    
    # Compute the Estimated Testing Set's Accuracy (Score),
    # for the Gaussian Naïve Bayes    
    gaussian_naive_bayes_true_test_accuracy = gaussian_naive_bayes.score(xs_test, ys_test)
    
    # Compute the Estimated Testing Error, regarding its Accuracy (Score),
    # for the Gaussian Naïve Bayes 
    gaussian_naive_bayes_true_test_error = ( 1 - gaussian_naive_bayes_true_test_accuracy )    
    
    
    # The Number of Samples, from the Testing Set 
    num_samples_test_set = len(xs_test)

    # The Number of Incorrect Predictions, regarding the Gaussian Naïve Bayes Classifier
    real_gaussian_naive_bayes_num_incorrect_predictions = 0
    
    # For each Sample, from the Testing Set
    for current_sample_test in range(num_samples_test_set):
        
        # If the Prediction/Classification of the Class for the current Sample,
        # of the Testing Set is different from the Real Class of the same,
        # it's considered an Real Error in Prediction/Classification,
        # regarding the Gaussian Naïve Bayes Classifier
        if(gaussian_naive_bayes_prediction_classes_for_samples_xs_test[current_sample_test] != ys_test[current_sample_test] ):
            real_gaussian_naive_bayes_num_incorrect_predictions += 1
            
    
    # Return the Predictions of the Samples,
    # the Real Number of Incorrect Prediction and the Estimated True/Testing Error,
    # in the Testing Set, for the Gaussian Naïve Bayes
    return gaussian_naive_bayes_prediction_classes_for_samples_xs_test, real_gaussian_naive_bayes_num_incorrect_predictions, gaussian_naive_bayes_true_test_error


# The Function to Perform the Classification Process for
# the Gaussian Naïve Bayes Classifier
def do_gaussian_naive_bayes():
    
    print("-----------------------------------------------------------------")
    print("3) Starting the Gaussian Naïve Bayes Classifier...")
    print("-----------------------------------------------------------------")
    print("\n\n")
    
    
    # The sum of the Training and Validation Errors, for Gaussian Naïve Bayes
    gaussian_naive_bayes_train_error_sum = 0
    gaussian_naive_bayes_valid_error_sum = 0
    
    # The K Folds Combinations Model, for the Stratified K Folds process
    k_folds = skl_model_selection.StratifiedKFold(n_splits = NUM_FOLDS)

        
    # The loop for all the combinations of K Folds, in the Stratified K Folds process
    for train_idx, valid_idx in k_folds.split(ys_train_classes, ys_train_classes):

        # Initialise the Gaussian Naïve Bayes Classifier
        gaussian_naive_bayes = skl_gaussian_naive_bayes()
        
        # Compute the Training and Validation Errors, for Gaussian Naïve Bayes
        gaussian_naive_bayes_train_error, gaussian_naive_bayes_valid_error = compute_gaussian_naive_bayes_errors(gaussian_naive_bayes, xs_train_features_std, ys_train_classes, train_idx, valid_idx)
        
        # Sum the current Training and Validation Errors to the Sums of them
        gaussian_naive_bayes_train_error_sum += gaussian_naive_bayes_train_error
        gaussian_naive_bayes_valid_error_sum += gaussian_naive_bayes_valid_error
        
        
    # Compute the Average of the Sums of the Training and Validation Errors, by the Total Number of Folds 
    gaussian_naive_bayes_train_error_avg_folds = ( gaussian_naive_bayes_train_error_sum / NUM_FOLDS )
    gaussian_naive_bayes_valid_error_avg_folds = ( gaussian_naive_bayes_valid_error_sum / NUM_FOLDS )
    
    
    # If the Boolean Flag for Debugging is set to True,
    # print some relevant information
    if(DEBUG_FLAG == True):   
        # Print the Training and Validation Errors
        print("\n")
        print("- Training Error = {}".format(gaussian_naive_bayes_train_error_avg_folds))
        print("- Validation Error = {}".format(gaussian_naive_bayes_valid_error_avg_folds))
        print("\n")

    # Compute the Predictions of the Samples, 
    # the Real Number of Incorrect Predictions and the Estimated True/Test Error,
    # of the Testing Set, for the Gaussian Naïve Bayes Classifier
    gaussian_naive_bayes_prediction_classes_for_samples_xs_test, real_gaussian_naive_bayes_num_incorrect_predictions, estimated_gaussian_naive_bayes_true_test_error = estimate_gaussian_naive_bayes_true_test_error(xs_train_features_std, ys_train_classes, xs_test_features_std, ys_test_classes)

    # If the Boolean Flag for Debugging is set to True,
    # print some relevant information
    if(DEBUG_FLAG == True):   
        # Print the Estimated True/Test Error
        print("\n")
        print("- Estimated True/Test Error = {}".format(estimated_gaussian_naive_bayes_true_test_error))
        print("- Number of Incorrect Predictions (Number of real Errors) = {}".format(real_gaussian_naive_bayes_num_incorrect_predictions))

    
    # The number of the Samples, from the Testing Set
    num_samples_test_set = len(xs_test_features_std)  

    # Computes the Aproximate Normal Test, for the Gaussian Naïve Bayes Classifier
    gaussian_naive_bayes_aproximate_normal_test_deviation_lower_bound, gaussian_naive_bayes_aproximate_normal_test_deviation_upper_bound = aproximate_normal_test(real_gaussian_naive_bayes_num_incorrect_predictions, estimated_gaussian_naive_bayes_true_test_error, num_samples_test_set)
    
    # If the Boolean Flag for Debugging is set to True,
    # print some relevant information
    if(DEBUG_FLAG == True):   
        # Print the Approximate Normal Test, with Confidence Level of 95% and
        # its Interval range of values, for the Test itself
        print("\n")
        print("- Approximate Normal Test, with Confidence Level of 95% = [ {} - {} ; {} + {} ]".format(real_gaussian_naive_bayes_num_incorrect_predictions, gaussian_naive_bayes_aproximate_normal_test_deviation_upper_bound, real_gaussian_naive_bayes_num_incorrect_predictions, gaussian_naive_bayes_aproximate_normal_test_deviation_upper_bound))
        print("- Approximate Normal Test Interval = [ {} ; {} ]".format( ( real_gaussian_naive_bayes_num_incorrect_predictions + gaussian_naive_bayes_aproximate_normal_test_deviation_lower_bound ) , ( real_gaussian_naive_bayes_num_incorrect_predictions + gaussian_naive_bayes_aproximate_normal_test_deviation_upper_bound ) ))
    
    
    # Return the Predictions of the Samples,
    # of the Testing Set, for the Gaussian Naïve Bayes Classifier
    return gaussian_naive_bayes_prediction_classes_for_samples_xs_test
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --------------------------------------------------------
# \                                                      \
# \  4) Comparing the Classifiers:                       \
# \     a) Logistic Regression,                          \
# \        varying the C Regularization Parameter        \
# \     b) Naïve Bayes,                                  \
# \        with custom KDEs (Kernel Density Estimations) \
# \        varying the Bandwidth Parameter               \
# \     c) Gaussian Naïve Bayes,                         \
# \        varying the Bandwidth Parameter               \
# \                                                      \
# \  - 4.1) Comparing by the Aproximate Normal Test      \
# \______________________________________________________\

def aproximate_normal_test(num_real_errors, probability_making_error, num_samples_test_set):

    #expected_num_errors = ( probability_making_error * num_samples_test_set )
    
    
    #probability_not_making_error = ( 1 - probability_making_error )
    
    #difference_num_errors = ( num_real_errors - expected_num_errors )
    #sqrt_expected_num_errors_mult_probability_not_making_error = mathematics.sqrt( expected_num_errors * probability_not_making_error )

    #aproximate_normal_test_value = ( difference_num_errors / sqrt_expected_num_errors_mult_probability_not_making_error )
    
    
    probability_occurrence_real_errors_in_test_set = ( num_real_errors / num_samples_test_set )
    
    probability_not_occurrence_real_errors_in_test_set = ( 1 - probability_occurrence_real_errors_in_test_set )
    
    
    aproximate_normal_test_std_deviation = mathematics.sqrt( num_samples_test_set * probability_occurrence_real_errors_in_test_set * probability_not_occurrence_real_errors_in_test_set )
    
    aproximate_normal_test_deviation_lower_bound = ( -1 * 1.96 * aproximate_normal_test_std_deviation )
    aproximate_normal_test_deviation_upper_bound = ( 1.96 * aproximate_normal_test_std_deviation )

        
    return aproximate_normal_test_deviation_lower_bound, aproximate_normal_test_deviation_upper_bound


# --------------------------------------------------------
# \                                                      \
# \  4) Comparing the Classifiers:                       \
# \     a) Logistic Regression,                          \
# \        varying the C Regularization Parameter        \
# \     b) Naïve Bayes,                                  \
# \        with custom KDEs (Kernel Density Estimations) \
# \        varying the Bandwidth Parameter               \
# \     c) Gaussian Naïve Bayes,                         \
# \        varying the Bandwidth Parameter               \
# \                                                      \
# \  - 4.2) Comparing by the McNemar's Test              \
# \______________________________________________________\

def mc_nemar_test(prediction_classes_for_samples_xs_test_classifier_1, prediction_classes_for_samples_xs_test_classifier_2):
    
    # The Number of Samples of the Testing Set
    num_samples_test_set = len(xs_test_features_std)
    
    samples_first_classifier_wrong_second_classifier_right_count = 0
    samples_first_classifier_right_second_classifier_wrong_count = 0
    
    for current_sample_test in range(num_samples_test_set):
        
        if( ( prediction_classes_for_samples_xs_test_classifier_1[current_sample_test] != ys_test_classes[current_sample_test] ) and ( prediction_classes_for_samples_xs_test_classifier_2[current_sample_test] == ys_test_classes[current_sample_test] ) ):
            samples_first_classifier_wrong_second_classifier_right_count += 1
        
        if( ( prediction_classes_for_samples_xs_test_classifier_1[current_sample_test] == ys_test_classes[current_sample_test] ) and ( prediction_classes_for_samples_xs_test_classifier_2[current_sample_test] != ys_test_classes[current_sample_test] ) ):
            samples_first_classifier_right_second_classifier_wrong_count += 1
    
    
    mc_nemar_test_dividend = ( ( abs(samples_first_classifier_wrong_second_classifier_right_count - samples_first_classifier_right_second_classifier_wrong_count) - 1) ** 2 )
    mc_nemar_test_divider = ( samples_first_classifier_wrong_second_classifier_right_count + samples_first_classifier_right_second_classifier_wrong_count )
    
    mc_nemar_test_value = ( mc_nemar_test_dividend / mc_nemar_test_divider )
    
    
    return mc_nemar_test_value


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ---- Run the 3 Classifiers: ------

# 1) Logistic Regression,
#    varying its Regularization C parameter    
logistic_regression_prediction_classes_for_samples_xs_test = do_logistic_regression()

print("\n\n")


# 2) Naïve Bayes, with customised KDE (Kernel Density Estimation),
#    varying its Bandwidth Regularization parameter
naive_bayes_prediction_classes_for_samples_xs_test = do_naive_bayes()

print("\n\n")


# 3) Gaussian Naïve Bayes
gaussian_naive_bayes_prediction_classes_for_samples_xs_test = do_gaussian_naive_bayes()

print("\n\n")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ---- Extra: McNemar Test Comparisons, between the Classifiers ------


print("-----------------------------------------------------------------")
print("EXTRA: McNemar Test Comparisons, between the Classifiers")
print("-----------------------------------------------------------------")
print("\n\n")


# a) McNemar Test #1: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs

print("Performing McNemar Test #1: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs...")

mc_nemar_test_logistic_regression_vs_naive_bayes_value = mc_nemar_test(logistic_regression_prediction_classes_for_samples_xs_test, naive_bayes_prediction_classes_for_samples_xs_test)

# If the Boolean Flag for Debugging is set to True,
# print some relevant information
if(DEBUG_FLAG == True):   
    # Print the result value for McNemar Test: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs
    print("\n")
    print("Result of the McNemar Test #1: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs:")
    print("- {}".format(mc_nemar_test_logistic_regression_vs_naive_bayes_value))
    
    # If the result value for McNemar Test: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs
    # is higher or equal than 3.84, with a Confidence Level of 95%
    if(mc_nemar_test_logistic_regression_vs_naive_bayes_value >= 3.84):
        
        # The Logistic Regression Classifier and Naïve Bayes Classifier, with custom KDEs, are significantly different
        print("\n")
        print("The Logistic Regression Classifier and Naïve Bayes Classifier, with custom KDEs, ARE significantly different!!!")
    
    # If the result value for McNemar Test: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs
    # is lower than 3.84, with a Confidence Level of 95%
    else:
    
        # The Logistic Regression Classifier and Naïve Bayes Classifier, with custom KDEs, are not significantly different
        print("\n")
        print("The Logistic Regression Classifier and Naïve Bayes Classifier, with custom KDEs, ARE NOT significantly different!!!")
        
    print("\n\n")
    
    
# b) McNemar Test #2: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier

print("\n\n")
print("Performing McNemar Test #2: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier...")

mc_nemar_test_logistic_regression_vs_gaussian_naive_bayes_value = mc_nemar_test(logistic_regression_prediction_classes_for_samples_xs_test, gaussian_naive_bayes_prediction_classes_for_samples_xs_test)

# If the Boolean Flag for Debugging is set to True,
# print some relevant information
if(DEBUG_FLAG == True):   
    # Print the result value for McNemar Test: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier
    print("\n")
    print("Result of the McNemar Test #2: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier:")
    print("- {}".format(mc_nemar_test_logistic_regression_vs_gaussian_naive_bayes_value))
    
    # If the result value for McNemar Test: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier
    # is higher or equal than 3.84, with a Confidence Level of 95%
    if(mc_nemar_test_logistic_regression_vs_gaussian_naive_bayes_value >= 3.84):
        
        # The Logistic Regression Classifier and Gaussian Naïve Bayes Classifier, are significantly different
        print("\n")
        print("The Logistic Regression Classifier and Gaussian Naïve Bayes Classifier, ARE significantly different!!!")
    
    # If the result value for McNemar Test: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier
    # is lower than 3.84, with a Confidence Level of 95%
    else:
    
        # The Logistic Regression Classifier and Gaussian Naïve Bayes Classifier, are not significantly different
        print("\n")
        print("The Logistic Regression Classifier and Gaussian Naïve Bayes Classifier, ARE NOT significantly different!!!")
        
    print("\n\n")
    
    
# c) McNemar Test #3: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier

print("\n\n")
print("Performing McNemar Test #3: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier...")

mc_nemar_test_naive_bayes_vs_gaussian_naive_bayes_value = mc_nemar_test(naive_bayes_prediction_classes_for_samples_xs_test, gaussian_naive_bayes_prediction_classes_for_samples_xs_test)

# If the Boolean Flag for Debugging is set to True,
# print some relevant information
if(DEBUG_FLAG == True):   
    # Print the result value for McNemar Test: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier
    print("\n")
    print("Result of the McNemar Test #3: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier:")
    print("- {}".format(mc_nemar_test_naive_bayes_vs_gaussian_naive_bayes_value))
    
    # If the result value for McNemar Test: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier
    # is higher or equal than 3.84, with a Confidence Level of 95%
    if(mc_nemar_test_naive_bayes_vs_gaussian_naive_bayes_value >= 3.84):
        
        # The Naïve Bayes Classifier, with custom KDEs and Gaussian Naïve Bayes Classifier, are significantly different
        print("\n")
        print("The Naïve Bayes Classifier, with custom KDEs and Gaussian Naïve Bayes Classifier, ARE significantly different!!!")
    
    # If the result value for McNemar Test: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier
    # is lower than 3.84, with a Confidence Level of 95%
    else:
    
        # The Naïve Bayes Classifier, with custom KDEs and Gaussian Naïve Bayes Classifier, are not significantly different
        print("\n")
        print("The Naïve Bayes Classifier, with custom KDEs and Gaussian Naïve Bayes Classifier, ARE NOT significantly different!!!")
        
    print("\n\n")
    


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~