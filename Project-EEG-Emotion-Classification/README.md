# EEG Signal Classification using Computational Intelligence

This repository contains the code and documentation for the course project on EEG signal classification using computational intelligence techniques. The project involves feature extraction, selection of effective features, classification, and evaluation of EEG signals to distinguish between positive and negative emotions using brain-computer interface (BCI) data.

## Overview

The goal of this project is to classify EEG signals into two emotional categories: **positive** and **negative**. EEG signals were collected from 10 individuals as they watched various videos designed to evoke emotions. Using feature extraction and classification techniques, we developed a model to distinguish between these two emotional states.

### Key Components:
- **EEG Data**: Recorded from 64 channels at a sampling rate of 1000 Hz.
- **Feature Extraction**: Extracted statistical and frequency domain features from the EEG signals.
- **Classification**: Implemented classifiers using **MLP** (Multi-Layer Perceptron) and **RBF** (Radial Basis Function) neural networks.
- **Evaluation**: K-fold cross-validation was used to evaluate the model's performance.

## Techniques Used

### Brain-Computer Interface (BCI)

BCI systems establish a communication link between the brain and external devices by interpreting EEG signals. In this project, we aim to control a classification system based on the emotional state detected through these signals.

### EEG Signal Processing

The EEG data was preprocessed to extract both **statistical features** (mean, variance, etc.) and **frequency domain features** (power spectral density) from each channel. These features are essential for distinguishing between the two emotional states.

### Neural Networks

Two types of neural networks were used to classify the EEG signals:
1. **Multi-Layer Perceptron (MLP)**: A feedforward neural network used for supervised learning tasks.
2. **Radial Basis Function (RBF) Network**: A neural network that uses radial basis functions as activation functions.

### Feature Extraction
   
   1. ** Statistical features **:
      Variance, Max abs, Kurtosis, Skewness
   
   3. ** Features in frequency domain**:
      Theta, Alpha, Lbeta, Mbeta, Hbeta, Gamma, Median freq, Mean freq, Entropy, OBW, Max power, Band power
### Feature Selection

Feature selection was done using Fisher's criterion, which evaluates the effectiveness of each feature in separating the two classes. The features with the highest separation capability were selected for classification.

### K-Fold Cross-Validation

We used **5-fold cross-validation** to ensure that the model generalizes well to unseen data. This method splits the data into 5 subsets, trains the model on 4, and tests it on the remaining one. The process is repeated until every subset has been used as a test set, and the final result is an average of all iterations.

## Dataset

The dataset consists of EEG recordings from 10 participants. The data was collected from 59 channels and recorded at a sampling rate of 1000 Hz. Each participant watched a series of videos, and EEG signals were recorded during the experiments.

- **Training Data**: 550 samples labeled as either positive or negative emotion.
- **Test Data**: 159 samples used to evaluate the model's performance.

### Data Format

- Each sample is represented by a 59x5000 matrix, where:
  - 59 = number of channels
  - 5000 = number of time points (representing 5 seconds of data)
  
Labels are either `1` (positive) or `-1` (negative).


## Contact

For any questions, feel free to contact Parnian Taheri at Parniantaheri.81@gmail.com.

   

