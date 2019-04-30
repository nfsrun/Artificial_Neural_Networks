# Artificial Neural Networks - Arabic Numerals to Computer Recognizable Integer Representations

## Introduction
The goal of this project was to use Artificial Neural Networks and make it have the ability to determine Arabic digits from 0-9 from grayscale images. 

## Algorithm
This artificial neural network utilized a backpropagation algorithm of one hidden layer with a sigmoid function as its activation function. 

## Data
Data was presented in a csv file where it was organized in a way where the first column contained the answer (integer value as a literal integer from 0-9) and the rest of the rows were grayscale values that formed the picture of a handwritten integer (organized row-column). There were 60000 test sample rows available on the csv file. 

## Test Steps
1. Randomize the number of test data for each specified size (line 6 of Program.java, the _sizes_ integer array) into a boolean array for keeping track of what is being tested on. 
2. Create a 2D array of selected test case rows to simulate the chosen randomized "test photos". 
3. Create an array of results by copying answer values of selected test sample rows over to a new array. 
4. Instantiate the neural network data structure with given hidden layer size (line 18 of ANNClassifier.java). 
5. Run the Algorithm: 
   5.a. Complete forward propagation using dot product, and the sigmoid function against each testing row sample. 
   5.b. Do backpropagation using the derivative of the sigmoid function and transposing dot products to get back to the hidden layer. 
   5.c. Repeat above two sub-steps for n times to simulate epochs. The value of n is equivalent to the epochMax variable value located in line 18 of ANNClassifier.java.  
6. Get the output of average accuracy percentage on the program console. 

## Variables (Changed one at a time in order)
### Hidden Units
Hidden units are the amount of "nodes" within the hidden layer of the artificial neural network (in this case the array size of the array simulating the hidden layer). The number of hidden units were set to 1 for data structure and algorithmic checking. Once I had confirmed that the data structure was functioning and the backpropagation algorithm was working as intended, hidden units were increased incrementally. I stopped at 33 hidden units. 
### Learning Rate
Variable used for the backpropagation algorithm that will stipulate on how fast the whole system will "learn" with gradient descent. I initially started at the instructor recommended rate of 0.01. I then tried tests by changing the learning rates with +-0.001. Both yielded accuracy that was worse than 0.01 and thus I kept it at 0.01. 
### Number of Test Data
This variable is important as very few pieces of data led to abysmal accuracy and too much data would lead to inefficient program performance (versus average accuracy gained). I tested this non-linearly with varying amounts of data. Based on the tests done, I deemed that at least 7000 random images had to be provided in order to get an approximate 90% average accuracy.
### Epochs
Epochs are the number of runs of the data to train the Artificial Neural Network Model. This too was a variable that also improves average accuracy as its value increases but at the expense of time. I have done tests that started from 13 epochs and slowly went up to 19. I deemed from my tests that 16 was the best epoch count to use at it provided accuracy that was not far off from 17-19 epochs, have greater accuracy improvement for smaller datasets on average (between 1-6000 given data samples) and was more time efficient to use.  

## Conclusion
# Average Accuracy: ~93%
