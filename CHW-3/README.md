## Computational Intelligence: Simulation - Series 3 Homework

**Due Date**: January 15, 1402 (Iranian calendar)

### Notes:
- Questions marked with a (*) are for extra credit.

---

### **Question 1**:
You are provided with the file `SampleData.mat`, which contains data for two classes with their corresponding labels.

1a) Plot the data of the two classes in 2D space.

1b) Split 30% of the data for validation and use the remaining data for training.

1c) Using a Radial Basis Function (RBF) neural network, classify the data. Determine the number of neurons in the hidden layer and the radius ($\sigma$) of the neurons to achieve the best validation results. You can use the `newrb` function in MATLAB.

Store the results, explanations, and plots in a single PDF file and the MATLAB code in a folder named `Ex1`.

---

### **Question 2** (Extra credit):
Write a program to implement the **k-means** clustering algorithm.

2a) The program should take the data and the number of clusters as input, and output the assigned clusters and initial cluster centers.

2b) Run the program on the `DataNew.mat` file (provided) and plot the results in 2D, visualizing the clusters and centers using different colors.

2c) Test the program for 4 and 6 clusters and compare the results.

2d) Compare your results with the output of MATLAB’s built-in `kmeans` function.

2e) Study and implement another clustering algorithm (e.g., **hierarchical clustering**, **LVQ**), and repeat parts (b) and (c) with this new method.

Store the code, results, and comparisons in a folder named `Ex2`.

---

### **Question 3**:
You are provided with the file `DataNew.mat`, which contains 1000 data points in 2D. We want to cluster the data into 5 clusters such that the **intra-cluster distance** is minimized (similar to a cost function in clustering).

Design and implement the following optimization algorithms for clustering:

3a) A genetic algorithm that clusters the data into at most 5 clusters.

3b) A genetic algorithm that clusters the data into exactly 5 clusters.

3c) A Particle Swarm Optimization (PSO) algorithm to cluster the data into at most 5 clusters.

3d) A PSO algorithm that clusters the data into exactly 5 clusters.

3e*) An Ant Colony Optimization (ACO) algorithm to cluster the data into at most 5 clusters (extra credit).

3f*) An ACO algorithm that clusters the data into exactly 5 clusters (extra credit).

Store the code and explanations in a folder named `Ex3`, and include the results and comparisons in a PDF file.
