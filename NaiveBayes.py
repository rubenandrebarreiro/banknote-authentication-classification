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

# Import the Kernel Density (Neighbors) Sub-Module,
# from SciKit-Learn Python's Library, as kernel_density
from sklearn.neighbors import KernelDensity as skl_kernel_density

# Import NumPy Python's Library as np
import numpy as np

# Import Accuracy Score (Metrics) Sub-Module,
# from SciKit-Learn Python's Library, as skl_accuracy_score
from sklearn.metrics import accuracy_score as skl_accuracy_score

# Import PyPlot Sub-Module, from Matplotlib Python's Library as plt
import matplotlib.pyplot as plt

# Import Model Selection Sub-Module, from SciKit-Learn Python's Library,
# as skl_model_selection 
from sklearn import model_selection as skl_model_selection


# The Number of Folds, for Stratified K Folds, in Cross-Validation
NUM_FOLDS = 5

# The Number of Steps/Variations for ajusting the Bandwidth parameter,
# for the Naïve Bayes
NUM_STEPS_BANDWIDTH_NAIVE_BAYES = 30

# -------------------------------------------------------
# \                                                     \
# \  Classifier 2) - Naïve Bayes,                       \
# \  with customised KDEs (Kernel Density Estimations), \
# \  varying its Bandwidth Regularization parameter     \
# \_____________________________________________________\

# The Function to compute the 
def compute_naive_bayes_errors(xs, ys, train_idx, valid_idx, bandwidth, num_classes, num_features):
    
    # Initialise the List of Logarithms of Base e of Prior Probabilities of
    # the Occurrence for each Class, in the Training Set,
    # for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    nb_logs_prior_prob_classes_occurrences_train_list = []
   
    kde = skl_kernel_density(bandwidth=bandwidth, kernel='gaussian')  # Initialise the Kernel Density Estimation (KDE) with current Bandwidth Regularization Parameter
    
    
    # In order to compute the Errors of the Naïve Bayes, it's needed to work with each pair of (Class, Feature) 
    
    # As, we have 2 classes and 4 features,  we will need a total of 8 Kernel Density Estimations (KDEs)
    # (2 Classes x 4 Features) = 8 Kernel Density Estimations)
    
    
    num_samples_all_xs_train = len(xs)               # The Number of Samples of the whole Training Set
    xs_train = xs[train_idx]                         # The Features of the Training Set
    ys_train = ys[train_idx]                         # The Classes of the Training Set
    num_samples_xs_train = len(xs_train)             # The Number of Samples of the Training Set
    ys_valid = ys[valid_idx]                         # The Classes of the Validation Set
    
    
    # The Logarithm Densities per each Class, for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    nb_log_densities_per_class = np.zeros((num_samples_all_xs_train, num_classes))                                
    
    # The Classifications/Predictions of the Samples of the whole Training Set, for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    nb_predict_all_xs_train_samples = np.zeros((num_samples_all_xs_train))

    
    # For each possible Class of the Dataset
    for current_class in range(num_classes):
    
        xs_train_current_class = xs_train[ys_train == current_class]             # The Samples of the Training Set, for the Current Class
        
        # Compute the Probabilities of the Occurrence for each Class, in the whole Training Set, for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
        nb_prior_prob_occurrences_for_current_class_train = ( len(xs_train[ys_train == current_class]) / num_samples_xs_train )
                
        # Compute the Logarithm of Base e of Prior Probabilities of the Occurrence for each Class, in the whole Training Set, to the respectively List for each Class, and append it to the respective List
        nb_logs_prior_prob_classes_occurrences_train_list.append( np.log(nb_prior_prob_occurrences_for_current_class_train) )
        
        # For each possible Feature of the Dataset
        for current_feature in range(num_features):
            
            # Fit the current Kernel Density Estimation (KDE), with the whole Training Set, for the current pair (Class, Feature), for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
            kde.fit(xs_train_current_class[:,[current_feature]])
            
            # Compute and sum the Logarithm Densities for the current pair (Class, Feature), for the current KDE (Kernel Density Estimation), for the whole Training Set
            nb_log_densities_per_class[:, current_class] += kde.score_samples(xs[:, [current_feature]])
        
        
        # Sum the Logarithm of Base e of Prior Probabilities of the Occurrence for each Class, in the whole Training Set,
        # to the Logarithm Densities per each Class, for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
        nb_log_densities_per_class[:, current_class] += nb_logs_prior_prob_classes_occurrences_train_list[current_class]

    
    # For each Sample of the whole Training Set, try to predict its Class
    for current_sample_x_all_xs_train in range(num_samples_all_xs_train):
        
        # Predict the current Sample of the whole Training Set, as the Maximum Argument (i.e., the Class) of it,
        # i.e. the argument/index with the highest probability of the Predictions of the Classes for each Sample
        nb_predict_all_xs_train_samples[current_sample_x_all_xs_train] = np.argmax( nb_log_densities_per_class[current_sample_x_all_xs_train] )
    
    
    nb_predict_xs_train_samples = nb_predict_all_xs_train_samples[train_idx]        # The Classifications/Predictions of the Samples of the Training Set, for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    
    nb_accuracy_train = skl_accuracy_score(ys_train, nb_predict_xs_train_samples)   # Compute the Accuracy of Score for the Predictions of the Classes for the Training Set  
   
    nb_error_train = ( 1 - nb_accuracy_train )                                      # Compute the Training Error, regarding the Accuracy Score for the Predictions of the Classes for the Training Set  

    nb_predict_xs_valid_samples = nb_predict_all_xs_train_samples[valid_idx]        # The Classifications/Predictions of the Samples of the Validation Set, for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
 
    nb_accuracy_valid = skl_accuracy_score(ys_valid, nb_predict_xs_valid_samples)   # Compute the Accuracy of Score for the Predictions of the Classes for the Validation Set 
    
    nb_error_valid = ( 1 - nb_accuracy_valid )                                      # Compute the Validation Error, regarding the Accuracy Score for the Predictions of the Classes for the Validation Set  
    
    
    # Return the Training and Validation Errors for the Naïve Bayes
    return nb_error_train, nb_error_valid
    

# The Function to Plot the Training and Validation, for the Naïve Bayes
def plot_train_valid_error_naive_bayes(train_error_values, valid_error_values):
    
    # Initialise the Plot
    plt.figure(figsize=(8, 8), frameon=True)

    # Set the line representing the continuous values,
    # for the Functions of the Training and Validation Errors
    plt.plot(train_error_values[:,0], train_error_values[:,1],'-', color="blue")
    plt.plot(valid_error_values[:,0], valid_error_values[:,1],'-', color="red")
    
    # Set the axis for the Plot
    plt.axis([min(valid_error_values[:,0]), max(valid_error_values[:,0]), min(train_error_values[:,1]), max(valid_error_values[:,1])])
    
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
def estimate_naive_bayes_true_test_error(xs_train, ys_train, xs_test, ys_test, best_bandwidth, num_classes, num_features):
    
    num_samples_xs_test = len(xs_test)                                              # The Number of Samples, in the Testing Set
    nb_logs_prior_prob_classes_occurrences_test_list = []                           # Initialise the List of Logarithms of Base e of Prior Probabilities of the Occurrence for each Class, in the Testing Set
    kde = skl_kernel_density(bandwidth=best_bandwidth, kernel='gaussian')           # Initialise the Kernel Density Estimation (KDE), with current Bandwidth Regularization Parameter   
    nb_log_densities_per_class_test = np.zeros((num_samples_xs_test, num_classes))  # The Logarithm Densities per each Class, in the Testing Set                           
    nb_predict_classes_xs_test = np.zeros((num_samples_xs_test))                    # The Classification/Prediction of the Samples, in the Testing Set, to the respective Classes
    

    # In order to compute the Errors of the Naïve Bayes, it's needed to work with each pair of (Class, Feature)
    # As, we have 2 classes and 4 features, we will need a total of 8 Kernel Density Estimations (KDEs)
    # (2 Classes x 4 Features) = 8 Kernel Density Estimations)
    
    # For each possible Class of the Dataset
    for current_class in range(num_classes):
        
        xs_train_current_class = xs_train[ys_train == current_class]
        
        # Compute the Probabilities of the Occurrence for each Class, in the Testing Set
        nb_prior_prob_occurrences_for_current_class_test = ( len(xs_train[ys_train == current_class]) / num_samples_xs_test )
                
        # Compute the Logarithm of Base e of Prior Probabilities of the Occurrence for each Class, in the Testing Set, to the respectively List for each Class, and append it to the respective List
        nb_logs_prior_prob_classes_occurrences_test_list.append( np.log(nb_prior_prob_occurrences_for_current_class_test) )
        

        # For each possible Feature of the Dataset
        for current_feature in range(num_features):
            
            # Fit the Kernel Density Estimation (KDE), with the Testing Set
            kde.fit(xs_train_current_class[:, current_feature].reshape(-1,1))
          
            # Compute and sum the Logarithm Densities for the current pair (Class, Feature), for the current KDE (Kernel Density Estimation), for the Testing Set
            nb_log_densities_per_class_test[:, current_class] += kde.score_samples(xs_test[:, [current_feature]])
        
        
        # Sum the Logarithm of Base e of Prior Probabilities of the Occurrence for each Class, in the Testing Set,
        # to the Logarithm Densities per each Class, for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
        nb_log_densities_per_class_test[:, current_class] += nb_logs_prior_prob_classes_occurrences_test_list[current_class]

    
    # For each Sample of the Testing Set, try to predict its Class
    for current_sample_x_test in range(num_samples_xs_test):
        
        # Predict the current Sample of the Testing Set, as the Maximum Argument (i.e., the Class) of it,
        # i.e. the argument/index with the highest probability of the Predictions of the Classes for each Sample, in the Testing Set
        nb_predict_classes_xs_test[current_sample_x_test] = np.argmax( nb_log_densities_per_class_test[current_sample_x_test] )
    
    
    # Compute the Accuracy of Score for the Predictions of the Classes for the Testing Set  
    nb_estimated_accuracy_test = skl_accuracy_score(ys_test, nb_predict_classes_xs_test)
    
    # Compute the Estimated True Testing Error, regarding the Accuracy Score for the Predictions of the Classes for the Testing Set  
    nb_estimated_true_error_test = ( 1 - nb_estimated_accuracy_test )
    
    
    # The Number of Samples, from the Testing Set 
    num_samples_test_set = len(xs_test)

    # The Real Number of Incorrect Predictions,
    # regarding the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    nb_num_incorrect_predictions = 0
    
    # For each Sample, from the Testing Set
    for current_sample_test in range(num_samples_test_set):
        
        # If the Prediction/Classification of the Class for the current Sample, of the Testing Set is different from the Real Class of the same,
        # it's considered an Real Error in Prediction/Classification, regarding the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
        if(nb_predict_classes_xs_test[current_sample_test] != ys_test[current_sample_test] ):
            nb_num_incorrect_predictions += 1
    
    
    # Return the Predictions of the Samples, the Real Number of Incorrect Predictions and the Estimated True/Test Error, or the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    return nb_predict_classes_xs_test, nb_num_incorrect_predictions, nb_estimated_true_error_test


def do_naive_bayes(ys_train_classes, xs_train_features_std, xs_test_features_std, ys_test_classes, num_classes, num_features):

    k_folds = skl_model_selection.StratifiedKFold(n_splits = NUM_FOLDS)        # The K Folds Combinations Model, for the Stratified K Folds process
    
    nb_train_error_values = np.zeros((NUM_STEPS_BANDWIDTH_NAIVE_BAYES, 2))     # The Values of Training and Validation Errors,
    nb_valid_error_values = np.zeros((NUM_STEPS_BANDWIDTH_NAIVE_BAYES, 2))     # for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    
    
    nb_best_bandwidth = 1e10                        # The Best Regularization Parameter Bandwidth found, for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    nb_best_valid_error_avg_folds = 1e10            # The Best Average of the Validation Error, for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    initial_bandwidth = 2e-2                        # The initial factor of each Bandwidth Step, for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    final_bandwidth = 6e-1                          # The final factor of each Bandwidth Step, for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    bandwidth_step = 2e-2                           # The factor of each Bandwidth Step, for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
    current_step_bandwidth_nb = 0                   # The Number of the current Bandwidth Step, for Naïve Bayes, with custom KDEs (Kernel Density Estimations)


    # The loop for try all the Regularization Parameter Bandwidths
    for current_bandwidth in np.arange(initial_bandwidth, ( final_bandwidth + bandwidth_step ), bandwidth_step):
        
        nb_train_error_sum = 0     # The sum of the Training and Validation Errors,
        nb_valid_error_sum = 0     # for Naïve Bayes, with custom KDEs (Kernel Density Estimations)

        
        # The loop for all the combinations of K Folds, in the Stratified K Folds process
        for train_idx, valid_idx in k_folds.split(ys_train_classes, ys_train_classes):
            
            # Compute the Training and Validation Errors, for Naïve Bayes, with custom KDEs (Kernel Density Estimations)
            nb_train_error, nb_valid_error = compute_naive_bayes_errors(xs_train_features_std, ys_train_classes, train_idx, valid_idx, current_bandwidth, num_classes, num_features)
            
            # Sum the current Training and Validation Errors to the Sums of them
            nb_train_error_sum += nb_train_error
            nb_valid_error_sum += nb_valid_error
            
            
        nb_train_error_avg_folds = ( nb_train_error_sum / NUM_FOLDS )                      # Compute the Average of the Sums of the 
        nb_valid_error_avg_folds = ( nb_valid_error_sum / NUM_FOLDS )                      #Training and Validation Errors, by the Total Number of Folds
            
       
        if(nb_best_valid_error_avg_folds > nb_valid_error_avg_folds):                      # Updates the Best Validation Error and also, the Best Regularization Bandwidth Parameter
            nb_best_valid_error_avg_folds = nb_valid_error_avg_folds
            nb_best_bandwidth = current_bandwidth
            
       
        nb_train_error_values[current_step_bandwidth_nb, 0] = current_bandwidth             # Store the Values for x and y, for all the Training Error values, 
        nb_train_error_values[current_step_bandwidth_nb, 1] = nb_train_error_avg_folds      #for the Plot of the Training Errors, as a Function of Logarithm of the Bandwidth Parameter

        nb_valid_error_values[current_step_bandwidth_nb, 0] = current_bandwidth             # Store the Values for x and y, for all the Validation Error values,
        nb_valid_error_values[current_step_bandwidth_nb, 1] = nb_valid_error_avg_folds      #for the Plot of the Validation Errors, as a Function of Logarithm of the Bandwidth Parameter
        

        current_step_bandwidth_nb += 1                                                      # Increment the Current Step of the Bandwidth Parameter value
    
        
    # Plot the Training and Validation Errors, for the Naïve Bayes Classifier
    plot_train_valid_error_naive_bayes(nb_train_error_values, nb_valid_error_values)
    
    # Compute the Predictions of the Samples, the Real Number of Incorrect Predictions and the Estimated True/Test Error,
    # for the Testing Set, of the Naïve Bayes Regression Classifier, with custom KDEs (Kernel Density Estimations)
    nb_predict_classes_xs_test, nb_num_incorrect_predict, estimated_nb_true_test_error = estimate_naive_bayes_true_test_error(xs_train_features_std, ys_train_classes, xs_test_features_std, ys_test_classes, nb_best_bandwidth, num_classes, num_features)    
    

    # Return the Predictions of the Samples, of the Testing Set, for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
    return nb_train_error_avg_folds, nb_valid_error_avg_folds, nb_best_bandwidth, nb_best_valid_error_avg_folds, nb_predict_classes_xs_test, nb_num_incorrect_predict, estimated_nb_true_test_error
