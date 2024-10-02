## Computational Intelligence: Series 1 Homework

**Due Date**: Friday, November 23

### Notes:
- Questions marked with a (*) carry extra credit.
- Please submit your code (with proper documentation) along with a report answering the provided questions.

---

**Question 1*:**

Consider a single neuron. Write a program that, for various input weights and thresholds, calculates the output of the neuron using the activation function $( f_{act} = \frac{1}{1+e^{-\beta(w_1\*x+w_2\*y-T)}})$
, where \( -1 < X, Y < 1 \). The program should:
- Plot the output in a 3D environment.
- Include interactive features to adjust the weights and thresholds.
- Plot the output when \(𝛽 = 𝑊1 = 𝑊2 = T = 1 \).

Implement and experiment with another common activation function (like OR). What happens when \( 𝛽 \) is set to a high value? What about when it’s set to a low value?

---

**Question 2:**

You are provided with a dataset named `iris.csv` that contains measurements from three species of iris flowers, including:
- Sepal length
- Sepal width
- Petal length
- Petal width

2-a) Separate two classes from the dataset based on any two features, and plot them in a 2D space. Which pair of features best separates the two classes? Justify your choice. 

2-b) Randomly select 5 samples from each class (for a total of 10 samples) and try to separate them using a TLU (Threshold Logic Unit). Provide a report detailing the separating hyperplane. 


2-c) Using 80% of the dataset for training (via both batch and online methods), train a TLU network.

2-d) plot the weight updates and thresholds for each epoch. Compare the performance of batch and online methods.

2-e) Report the accuracy and final weights/thresholds for each method. 

2-f) Also, test the model on the remaining 20% of the dataset and report the results.

2-g) In a 3D space, separate three classes using a separating hyperplane. If this is not feasible using a TLU, explain why and suggest alternatives for more complex classification.
