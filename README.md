# deep-learning-challenge
Module 21 challenge


# Report on the Performance of the Deep Learning Model for Alphabet Soup

## Overview of the Analysis:
The purpose of this analysis is to develop a deep learning model using TensorFlow and Keras to predict which applicants have the best chances of success if funded by Alphabet Soup, a nonprofit organization. The goal is to create a predictive tool using a training model based on historical data on successful and unsuccessful funding applications, so that this tool can identify potentially successful applicants, thus optimizing the allocation of resources.

## Results:

### Data Preprocessing:
* **Target Variable(s):**
  * IS_SUCCESSFUL: Indicates whether the funding application was successful (1) or not (0).
* **Feature Variables:**
  * All other columns except EIN and NAME, which were removed.
* **Variables Removed:**
  * EIN and NAME were irrelevant for this training model.

### Compiling, Training, and Evaluating the Model:
* **Neurons, Layers, and Activation Functions:**
  * Input Layer:
    * Neurons: 100
    * Activation Function: Swish
  * Hidden Layers (Additional 5 layers):
    * Neurons: 80, 60, 60, 60, 60
    * Activation Function: ReLU
  * Output Layer:
    * Neurons: 1
    * Activation Function: Sigmoid

* **Target Model Performance:**
  * The target model performance of 75% and above was not achieved. After 10 attempts, the highest accuracy obtained was approximately 73%.

* **Steps to Increase Model Performance:**
  * This process took easily 10 tries and only gained 1% extra in the accuracy of the model from the original AlphabetSoupCharity_ModelStarterCode.ipynb file.
    
    * First, features that were deemed unnecessary were removed.
    * Data from other features were binned, and cutoff values were created.
    * Different activations were used during training, such as ReLU, LeakyReLU, SELU, ELU, Swish, and Tanh, in various combinations, some yielding worse results than others.
    * The number of units in the input layer was increased from 80 to 100. For the hidden layers, 80 and 60 were the numbers tested to see what conveyed the best performance.
    * In some versions, more layers were added.
    * L2 regularization with a coefficient of 0.001 was implemented.
    * Various optimizers were used when compiling the model, including Adam, SGD, RMSprop, and Nadam. Adam was the top performer, followed by SGD.
    * Early stopping, learning rate scheduling, and model checkpointing were employed to optimize training.
    * Numerous combinations were tried regarding the number of neurons, the activations used, the number of layers, and optimizers. Versions without removing features and with different binning strategies were also tested.

## Summary:
This deep learning model achieved the highest accuracy of about 73% in predicting the success of applicants funded by Alphabet Soup. Despite various optimization attempts, including altering activation functions, adjusting network architecture, and changing optimizers, improvement in performance was minimal, with accuracy staying between 72% and 73%.

I am sure that by spending more time with the dataset and creating diverse combinations of the above-mentioned methods, a better outcome could potentially be achieved in due time, especially with future advancements in predictability with deep learning models. For the moment, these were the highest results obtained.






References:
Using Keras for Machine Learning:
https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/

Saving and Loading Models: 
https://colab.research.google.com/github/agungsantoso/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%206%20-%20Saving%20and%20Loading%20Models.ipynb
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_load.ipynb
https://stackoverflow.com/questions/64808087/how-do-i-save-files-from-google-colab-to-google-drive
https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory
