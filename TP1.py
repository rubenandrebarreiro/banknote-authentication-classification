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

from GaussianNaiveBayes import do_gaussian_naive_bayes;

from LogisticRegression import do_logistic_regression;

from NaiveBayes import do_naive_bayes


# Definition of the necessary Python Libraries

# a) General Libraries:

# Import NumPy Python's Library as np
import numpy as np

# Import Math Python's Library as mathematics
import math as mathematics


# Import SciKit-Learn as skl
import sklearn as skl



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Constants #1


# The Number of Folds, for Stratified K Folds, in Cross-Validation
NUM_FOLDS = 5

# The Number of Steps/Variations for ajusting the C Regularization parameter, for the Logistic Regression
NUM_STEPS_C_REGULARIZATION_LOGISTIC_REGRESSION = 15

# The Number of Steps/Variations for ajusting the Bandwidth parameter, for the Naïve Bayes
NUM_STEPS_BANDWIDTH_NAIVE_BAYES = 30

# The Boolean Flag for Debugging
DEBUG_FLAG = True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The files of the Datasets for Training and Testing


train_set_data_file = "files/data/TP1_train.tsv"                               # The Data for Training Set
test_set_data_file = "files/data/TP1_test.tsv"                                 # The Data for Testing Set


train_set_data_not_random = np.loadtxt(train_set_data_file, delimiter="\t")    # Load the Data for Training Set with NumPy function loadtxt
test_set_data_not_random = np.loadtxt(test_set_data_file, delimiter="\t")      # Load the Data for Testing Set with NumPy function loadtxt

train_set_data_random = skl.utils.shuffle(train_set_data_not_random)           # Shuffle the Training Set, not randomized
test_set_data_random = skl.utils.shuffle(test_set_data_not_random)             # Shuffle the Testing Set, not randomized

ys_train_classes = train_set_data_random[:,-1]                                 # Select the Classes of the Training Set, randomized
ys_test_classes = test_set_data_random[:,-1]                                   # Select the Classes of the Testing Set, randomized

xs_train_features = train_set_data_random[:,0:-1]                              # Select the Features of the Training Set, randomized
xs_test_features = test_set_data_random[:,0:-1]                                # Select the Features of the Testing Set, randomized

test_set_size = len(xs_test_features)                                          # The size of the Data for Testing Set, randomized

train_means = np.mean(xs_train_features,axis=0)                                # Computing the Means of the Training Set, randomized
train_stdevs = np.std(xs_train_features,axis=0)                                # Computing the Standard Deviations of the Training Set, randomized

xs_train_features_std = ( ( xs_train_features - train_means ) / train_stdevs ) # Standardize the Training Set, randomized
xs_test_features_std = ( ( xs_test_features - train_means ) / train_stdevs )   # Standardize the Testing Set, randomized



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Constants #2


# The Number of Features (i.e., 4 Features, per each Banknote)
NUM_FEATURES = xs_train_features_std.shape[1]

# The Number of Classes (i.e., 2 Classes possible, per each Banknote, Real or Fake)
NUM_CLASSES = len(set(ys_train_classes))

num_samples_test_set = len(xs_test_features_std) 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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

def aproximate_normal_test(num_real_errors, prob_making_error, num_samples_test_set):

    prob_errors_in_test_set = ( num_real_errors / num_samples_test_set )
    prob_not_errors_in_test_set = ( 1 - prob_errors_in_test_set )

    NormalTest_deviation = mathematics.sqrt( num_samples_test_set * prob_errors_in_test_set * prob_not_errors_in_test_set )
    
    NormalTest_LowerDeviation = ( -1 * 1.96 * NormalTest_deviation )
    NormalTest_UpperDeviation = ( 1.96 * NormalTest_deviation )

    return NormalTest_LowerDeviation, NormalTest_UpperDeviation


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

def mc_nemar_test(predict_classes_xs_test_1, predict_classes_xs_test_2):
    
    num_samples_test_set = len(xs_test_features_std)
    
    first_wrong_second_right = 0
    first_right_second_wrong = 0
    
    for current_sample_test in range(num_samples_test_set):
        
        if( ( predict_classes_xs_test_1[current_sample_test] != ys_test_classes[current_sample_test] ) and ( predict_classes_xs_test_2[current_sample_test] == ys_test_classes[current_sample_test] ) ):
            first_wrong_second_right += 1
        
        if( ( predict_classes_xs_test_1[current_sample_test] == ys_test_classes[current_sample_test] ) and ( predict_classes_xs_test_2[current_sample_test] != ys_test_classes[current_sample_test] ) ):
            first_right_second_wrong += 1
    
    
    mc_nemar_test_dividend = ( ( abs(first_wrong_second_right - first_right_second_wrong) - 1) ** 2 )
    mc_nemar_test_divider = ( first_wrong_second_right + first_right_second_wrong )
    
    mc_nemar_test_value = ( mc_nemar_test_dividend / mc_nemar_test_divider )
    
    
    return mc_nemar_test_value



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ---- Run the 3 Classifiers: ------




# 1) Computes the Logistic Regression Classifier
logReg_train_error_avg_folds, logReg_valid_error_avg_folds, logReg_best_c, logReg_best_valid_error_avg_folds, logReg_predict_classes_xs_test, logReg_num_incorrect_predictions, estimated_logReg_true_test_error = do_logistic_regression(xs_test_features_std, ys_train_classes, xs_train_features_std, ys_test_classes, NUM_FEATURES)

# Computes the Aproximate Normal Test, for the Logistic Regression Classifier
logReg_normalTest_LowDeviation, logReg_normalTest_UpperDeviation = aproximate_normal_test(logReg_num_incorrect_predictions, estimated_logReg_true_test_error, num_samples_test_set)


# 2) Computes the Naive Bayes Classifier
nb_train_error_avg_folds, nb_valid_error_avg_folds, nb_best_bandwidth, nb_best_valid_error_avg_folds, nb_predict_classes_xs_test, nb_num_incorrect_predict, estimated_nb_true_test_error = do_naive_bayes(ys_train_classes, xs_train_features_std, xs_test_features_std, ys_test_classes, NUM_CLASSES, NUM_FEATURES)

# Computes the Aproximate Normal Test, for the Naïve Bayes Classifier, with custom KDEs (Kernel Density Estimations)
nb_NormalTest_LowerDeviation, nb_NormalTest_UpperDeviation = aproximate_normal_test(nb_num_incorrect_predict, estimated_nb_true_test_error, num_samples_test_set)
 

# 3) Computes the Gaussian Naive Bayes Classifier
gnb_train_error_avg_folds, gnb_valid_error_avg_folds, gnb_predict_classes_xs_test, gnb_num_incorrect_predict, estimated_gnb_true_test_error = do_gaussian_naive_bayes(ys_train_classes, xs_train_features_std, xs_test_features_std, ys_test_classes)

# Computes the Aproximate Normal Test, for the Gaussian Naïve Bayes Classifier
gnb_NormalTest_LowerDeviation, gnb_NormalTest_UpperDeviation = aproximate_normal_test(gnb_num_incorrect_predict, estimated_gnb_true_test_error, num_samples_test_set)
  

#--------------- Logistic Regression -------------------------------------#
print("-----------------------------------------------------------------")
print("1) Starting the Logistic Regression Classifier...")
print("-----------------------------------------------------------------")

# Print the Training and Validation Errors
print("\n")
print("- Training Error = {}".format(logReg_train_error_avg_folds))
print("- Validation Error = {}".format(logReg_valid_error_avg_folds))

# Print the Best Value for the Regularization C Parameter
print("\n")
print("Best Value for Regularization C = {} :".format(logReg_best_c))
print("- Best Validation Error = {}".format(logReg_best_valid_error_avg_folds))

# Print the Estimated True/Test Error
print("\n")
print("- Estimated True/Test Error = {}".format(estimated_logReg_true_test_error))
print("- Number of Incorrect Predictions (Number of real Errors) = {}".format(logReg_num_incorrect_predictions))

# Print the Approximate Normal Test, with Confidence Level of 95% and its Interval range of values, for the Test itself
print("\n")
print("- Approximate Normal Test, with Confidence Level of 95% = [ {} - {} ; {} + {} ]".format(logReg_num_incorrect_predictions, logReg_normalTest_UpperDeviation, logReg_num_incorrect_predictions, logReg_normalTest_UpperDeviation))
print("- Approximate Normal Test Interval = [ {} ; {} ]".format( ( logReg_num_incorrect_predictions + logReg_normalTest_LowDeviation ) , ( logReg_num_incorrect_predictions + logReg_normalTest_UpperDeviation ) ))
print("\n\n")
#--------------- Logistic Regression -------------------------------------#




#--------------- Naive Bayes ---------------------------------------------#
print("-----------------------------------------------------------------")
print("2) Starting the Naïve Bayes Classifier...")
print("-----------------------------------------------------------------")

# Print the Training and Validation Errors
print("\n")
print("- Training Error = {}".format(nb_train_error_avg_folds))
print("- Validation Error = {}".format(nb_valid_error_avg_folds))

# Print the Best Value for the Regularization Bandwidth Parameter
print("\n")
print("Best Value for Regularization Bandwidth = {} :".format(nb_best_bandwidth))
print("- Best Validation Error = {}".format(nb_best_valid_error_avg_folds))

# Print the Estimated True/Test Error
print("\n")
print("- Estimated True/Test Error = {}".format(estimated_nb_true_test_error))
print("- Number of Incorrect Predictions (Number of real Errors) = {}".format(nb_num_incorrect_predict))

# Print the Approximate Normal Test, with Confidence Level of 95% and its Interval range of values, for the Test itself
print("\n")
print("- Approximate Normal Test, with Confidence Level of 95% = [ {} - {} ; {} + {} ]".format(nb_num_incorrect_predict, nb_NormalTest_UpperDeviation, nb_num_incorrect_predict, nb_NormalTest_UpperDeviation))
print("- Approximate Normal Test Interval = [ {} ; {} ]".format( ( nb_num_incorrect_predict + nb_NormalTest_LowerDeviation ) , ( nb_num_incorrect_predict + nb_NormalTest_UpperDeviation ) ))
print("\n\n")
#--------------- Naive Bayes ---------------------------------------------#




#--------------- Gaussian Naive Bayes ------------------------------------#
print("-----------------------------------------------------------------")
print("3) Starting the Gaussian Naïve Bayes Classifier...")
print("-----------------------------------------------------------------")

# Print the Training and Validation Errors
print("\n")
print("- Training Error = {}".format(gnb_train_error_avg_folds))
print("- Validation Error = {}".format(gnb_valid_error_avg_folds))
      
# Print the Estimated True/Test Error
print("\n")
print("- Estimated True/Test Error = {}".format(estimated_gnb_true_test_error))
print("- Number of Incorrect Predictions (Number of real Errors) = {}".format(gnb_num_incorrect_predict))

# Print the Approximate Normal Test, with Confidence Level of 95% and its Interval range of values, for the Test itself
print("\n")
print("- Approximate Normal Test, with Confidence Level of 95% = [ {} - {} ; {} + {} ]".format(gnb_num_incorrect_predict, gnb_NormalTest_UpperDeviation, gnb_num_incorrect_predict, gnb_NormalTest_UpperDeviation))
print("- Approximate Normal Test Interval = [ {} ; {} ]".format( ( gnb_num_incorrect_predict + gnb_NormalTest_LowerDeviation ) , ( gnb_num_incorrect_predict + gnb_NormalTest_UpperDeviation ) ))
print("\n\n")
#--------------- Gaussian Naive Bayes ------------------------------------#




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ---- Extra: McNemar Test Comparisons, between the Classifiers ------

mc_nemar_test_logistic_regression_vs_naive_bayes_value = mc_nemar_test(logReg_predict_classes_xs_test, nb_predict_classes_xs_test)
mc_nemar_test_logistic_regression_vs_gaussian_naive_bayes_value = mc_nemar_test(logReg_predict_classes_xs_test, gnb_predict_classes_xs_test)
mc_nemar_test_naive_bayes_vs_gaussian_naive_bayes_value = mc_nemar_test(nb_predict_classes_xs_test, gnb_predict_classes_xs_test)


print("-----------------------------------------------------------------")
print("EXTRA: McNemar Test Comparisons, between the Classifiers")
print("-----------------------------------------------------------------")
print("\n\n")



#-----------------------------------------Logistic Regression vs Naive Bayes ------------------------------------------------------#
# a) McNemar Test #1: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs

print("Performing McNemar Test #1: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs...")

# Print the result value for McNemar Test: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs
print("\n")
print("Result of the McNemar Test #1: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs:")
print("- {}".format(mc_nemar_test_logistic_regression_vs_naive_bayes_value))
    
# If the result value for McNemar Test: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs is higher or equal than 3.84, with a Confidence Level of 95%
if(mc_nemar_test_logistic_regression_vs_naive_bayes_value >= 3.84):
        
    # The Logistic Regression Classifier and Naïve Bayes Classifier, with custom KDEs, are significantly different
    print("\n")
    print("The Logistic Regression Classifier and Naïve Bayes Classifier, with custom KDEs, ARE significantly different!!!")
    
# If the result value for McNemar Test: Logistic Regression Classifier vs. Naïve Bayes Classifier, with custom KDEs is lower than 3.84, with a Confidence Level of 95%
else:
    
    # The Logistic Regression Classifier and Naïve Bayes Classifier, with custom KDEs, are not significantly different
    print("\n")
    print("The Logistic Regression Classifier and Naïve Bayes Classifier, with custom KDEs, ARE NOT significantly different!!!")
#-----------------------------------------Logistic Regression vs Naive Bayes ------------------------------------------------------#
   
 
   
 
#-----------------------------------------Logistic Regression vs Gaussian Naive Bayes ---------------------------------------------#
# b) McNemar Test #2: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier

print("\n\n")
print("Performing McNemar Test #2: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier...")

 
# Print the result value for McNemar Test: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier
print("\n")
print("Result of the McNemar Test #2: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier:")
print("- {}".format(mc_nemar_test_logistic_regression_vs_gaussian_naive_bayes_value))
    
# If the result value for McNemar Test: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier is higher or equal than 3.84, with a Confidence Level of 95%
if(mc_nemar_test_logistic_regression_vs_gaussian_naive_bayes_value >= 3.84):
        
    # The Logistic Regression Classifier and Gaussian Naïve Bayes Classifier, are significantly different
    print("\n")
    print("The Logistic Regression Classifier and Gaussian Naïve Bayes Classifier, ARE significantly different!!!")
    
# If the result value for McNemar Test: Logistic Regression Classifier vs. Gaussian Naïve Bayes Classifier is lower than 3.84, with a Confidence Level of 95%
else:
    
   # The Logistic Regression Classifier and Gaussian Naïve Bayes Classifier, are not significantly different
   print("\n")
   print("The Logistic Regression Classifier and Gaussian Naïve Bayes Classifier, ARE NOT significantly different!!!")
#-----------------------------------------Logistic Regression vs Gaussian Naive Bayes ---------------------------------------------#
  


#-----------------------------------------Naive Bayes vs Gaussian Naive Bayes -----------------------------------------------------#  
# c) McNemar Test #3: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier

print("\n\n")
print("Performing McNemar Test #3: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier...")
 
# Print the result value for McNemar Test: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier
print("\n")
print("Result of the McNemar Test #3: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier:")
print("- {}".format(mc_nemar_test_naive_bayes_vs_gaussian_naive_bayes_value))
    
# If the result value for McNemar Test: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier is higher or equal than 3.84, with a Confidence Level of 95%
if(mc_nemar_test_naive_bayes_vs_gaussian_naive_bayes_value >= 3.84):
        
    # The Naïve Bayes Classifier, with custom KDEs and Gaussian Naïve Bayes Classifier, are significantly different
    print("\n")
    print("The Naïve Bayes Classifier, with custom KDEs and Gaussian Naïve Bayes Classifier, ARE significantly different!!!")
    
# If the result value for McNemar Test: Naïve Bayes Classifier, with custom KDEs vs. Gaussian Naïve Bayes Classifier is lower than 3.84, with a Confidence Level of 95%
else:
    
    # The Naïve Bayes Classifier, with custom KDEs and Gaussian Naïve Bayes Classifier, are not significantly different
    print("\n")
    print("The Naïve Bayes Classifier, with custom KDEs and Gaussian Naïve Bayes Classifier, ARE NOT significantly different!!!")   
#-----------------------------------------Naive Bayes vs Gaussian Naive Bayes -----------------------------------------------------#  


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~