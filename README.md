## AlphabetSoup Charity Optimization

This repository contains notebooks for optimizing a deep learning model to predict the success of charity donations using TensorFlow and Keras.

## Notebooks

1. AlphabetSoupCharity.ipynb
   -------
   
### Purpose

This notebook presents the initial deep learning model for predicting successful charity donations.

### Key Steps

- Data preprocessing: Encodes categorical variables (APPLICATION_TYPE, CLASSIFICATION) using one-hot encoding.

- Feature selection: Removes non-beneficial ID columns (EIN, NAME).

- Model configuration: Builds a basic deep neural network with default layer configurations.

- Model training: Fits the model and evaluates its performance.

### Results

- Achieved an accuracy of approximately 72.54% on the test dataset.

- Provides a baseline model for comparison with the optimized model.
  
  
2. AlphabetSoupCharity_Optimization.ipynb
   ---

   
### Purpose

This notebook focuses on optimizing the deep learning model for predicting successful charity donations.

### Key Steps:

- Data preprocessing: Includes binning and encoding categorical variables (APPLICATION_TYPE, CLASSIFICATION).

- Feature selection: Removes non-beneficial ID columns (EIN, STATUS, SPECIAL_CONSIDERATIONS).

- Model configuration: Builds a deep neural network with optimized layers and neurons.

- Model training: Uses scaled training data to fit and evaluate the model.

### Results:

- Achieved an accuracy of approximately 78.17% on the test dataset.

- Implemented binning and preprocessing techniques to improve model performance.



## Usage

1. Open and run the notebooks using Jupyter Notebook or Google Colab

2. View the optimized model file (AlphabetSoupCharity_Optimization.h5) and baseline model file (alphabet_model.h5) in the repository.

-------------------------------------------------------------------

# Analysis

## Final Report: Optimizing a Deep Learning Model for Charity Donations

## Overview of the Analysis

The purpose of this analysis is to develop and optimize a deep learning model using TensorFlow and Keras to predict whether applicants will be successful if funded by Alphabet Soup. By analyzing and preprocessing the data, the aim is to train a model that can effectively classify successful and unsuccessful applicants.

## Results

### Data Preprocessing

- Target Variable:

The target variable for the model is “IS_SUCCESSFUL”, which indicates whether the applicant was successful (1) or not (0) in receiving funding.

- Feature Variables:

The features used for the model include all columns from the dataset after preprocessing excluding the target variable. This includes Name, Application Type, Affiliation, Special Considerations, Classification, Use Case, Organization, Income Amount, and Asking Amount

- Variables Removed:
  
The EIN (Employer Identification Number) and Loan Status were removed from the input data as they are not relevant for predicting success.

### Compiling, Training, and Evaluating the Model

Neural Network Configuration:

The neural network model consists of three layers:

1. The first hidden layer has 90 neurons with a ReLU activation function.

2. The second hidden layer has 20 neurons with a ReLU activation function.

3. The output layer uses a sigmoid activation function for binary classification.

Model Training and Performance:

- The model was trained using the Adam optimizer and binary cross-entropy loss function.

- The model was trained for 100 epochs, achieving an accuracy of approximately 77.96% on the test dataset.

Performance Optimization Attempts:

- Binning and preprocessing of categorical variables (APPLICATION_TYPE, CLASSIFICATION, and NAME) were performed to reduce dimensionality and enhance model performance.

- Adjustments made to the neurons in the layers in order to achieve better results.

- Removal of EIN and Status to remove noise.

### Summary

In summary, the optimized deep learning model achieved an accuracy of 78.17% in predicting whether applicants would be successful in receiving funding from Alphabet Soup. Using ReLU with two hidden layers, removing noise, and binning Classification, Name, and Application Types help the model perform more efficiently. More work can be done to train the model in order to achieve a higher optimization and accuracy rate, including testing of different models, or the introduction of more data.
