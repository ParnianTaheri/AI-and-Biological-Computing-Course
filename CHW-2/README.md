## Computational Intelligence: Simulation - Series 2 Homework

**Due Date**: December 7, 1402 (Iranian calendar)

### Notes:
- Questions 1 and 2 can be implemented in Python.
- **Question 3 is optional for extra credit**.

---

### **Question 1**:
You are provided with the file `Ex1.mat`, which contains three vectors: fuel rate, speed, and nitrogen oxide (NOx) emission. The goal is to estimate NOx emissions based on fuel rate and speed.

1a) Use the `scatter3` command to plot the output variable based on the two input variables.

1b) Treat the first 700 rows as training data and the rest as validation data (selected randomly).

1c) Using linear regression, fit a model to the training data and calculate the **Mean Square Error (MSE)** for both training and validation sets.
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

1d) Use logistic regression to fit another model and calculate the **MSE** on both training and validation sets.

1e) Use `nnstart` or the `fitnet` toolbox to create a neural network with one hidden layer (MLP) and fit it to the training data. Adjust the number of neurons in the hidden layer to minimize the validation MSE.

Place all results, including explanations and plots, in a single PDF file, and store the MATLAB code and results in a folder named `Ex1`.

---

### **Question 2**:
You are provided with the file `Ex2.mat`, containing a dataset with two classes (0 and 1). Each data sample is represented by a 3-dimensional vector.

- `TrainData` contains 90 samples (with the first 3 columns being features and the last column being the class label).
- `TestData` contains 90 samples with unknown labels.

2a) Use a perceptron network to classify the training data in two scenarios:
   - Case 1: The output neuron has one hidden layer, where the output determines class membership (0 or 1).
   - Case 2: The output neuron has two hidden layers. For a given data sample, if the first output neuron is 0.7 and the second is 0.3, it belongs to class 1.

2b) Plot the data for each class in 3D space using different colors to visualize the separation between the classes. Use 20% of the data for validation.

2c) Test various configurations of the number of hidden neurons and report the classification accuracy on the validation data. Finally, apply the model to the test data.

Store all results and MATLAB code in a folder named `Ex2`.

---

### **Question 3** (Optional):
An example code for a simple MLP network using **Error Backpropagation** is provided (`MLP_Example.ipynb`). Your task is to complete the missing sections and experiment with different activation functions (`logistic`, `tanh`).

3a) Complete the derivatives of the activation functions in the code.

3b) Define the input-output structure of the network for the XOR problem.

3c) Run the backpropagation algorithm and analyze the outputs (e.g., error values, deltas).

3d) Print the error values after each epoch, and plot the changes in error as a function of epoch count.

Store the final code and results in a folder named `Ex3`.
