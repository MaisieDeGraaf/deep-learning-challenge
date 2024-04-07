## AlphabetSoup Charity Optimization

This repository contains notebooks for optimizing a deep learning model to predict the success of charity donations using TensorFlow and Keras.

## Notebooks
1. AlphabetSoupCharity_Optimization.ipynb
   
### Purpose

This notebook focuses on optimizing the deep learning model for predicting successful charity donations.

### Key Steps:

- Data preprocessing: Includes binning and encoding categorical variables (APPLICATION_TYPE, CLASSIFICATION).

- Feature selection: Removes non-beneficial ID columns (EIN, STATUS, SPECIAL_CONSIDERATIONS).

- Model configuration: Builds a deep neural network with optimized layers and neurons.

- Model training: Uses scaled training data to fit and evaluate the model.

### Results:

- Achieved an accuracy of approximately 78.13% on the test dataset.

- Implemented binning and preprocessing techniques to improve model performance.

2. AlphabetSoupCharity.ipynb
   
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

## Usage

1. Open and run the notebooks using Jupyter Notebook or Google Colab

2. View the optimized model file (AlphabetSoupCharity_Optimization.h5) and baseline model file (alphabet_model.h5) in the repository.

# Analysos
