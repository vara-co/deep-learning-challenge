<h1 align="center">DU - University of Denver<br/>
Data Analysis & Visualization Bootcamp<br/></h1>

--------------------------------

<h2 align="center">Deep Learning Models<br/>
Module 21 Challenge
<br/>
By Laura Vara</h2><br/>

(IMAGE) 

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

---------------------------------
INDEX
---------------------------------
1. Overview of the Analysis, Results, and Summary (Above)
2. Content of the repository
3. Instructions for the Project
4. References

---------------------------------
Content of the repository
---------------------------------
- nn_models1 directory:
  - AlphabetSoupCharity.h5  <-- model for the AlphabetSoupCharity_ModelStarterCode.ipynb
  - AlphabetSoupCharity_Optimization.h5 <-- the best performing model
  - AlphabetSoupCharity_mod2.h5
  - AlphabetSoupCharity_mod3.h5
  - AlphabetSoupCharity_mod4.h5
  - AlphabetSoupCharity_mod5.h5
  - lending_data.csv

- AlphabetSoupCharity_ModelStarterCode.ipynb <-- first model worked on with 72% accuracy
- AlphabetSoupCharity_Optimization.ipynb <-- last model with the highest accuracy of 73%
- Starter_Code.ipynb <-- starter file provided to create the model.
----------------------------------
Instructions
----------------------------------

The instructions for this Challenge are divided into the following subsections:
* Step 1: Preprocess the Data
* Step 2: Compile, Train, and Evaluate the Model
* Step 3: Optimize the Model
* Step 4: Write a Report on the Neural Network Model
* Step 5: Add the files to your Repo

### 1. Preprocess the Data
Using your knowledge of Pandas and scikit-learn's **StandardScaler()**, you'll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.
- Start by uploading the starter file to Google Colab, then using the information provided in the Challenge files, follow the instructions to complete the preprocessing steps.
  1. Read in the **charity_data.csv** to a Pandas DataFrame, and be sure to identify the following in your dataset:
     - What variable(s) are the target(s) for your model?
     - What variable(s) are the feature(s) for your model?
    
  2. Drop the **EIN** and **NAME** columns
  3. Determine the number of unique values for each column.
  4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
  5. Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, **other**, and then check if the replacement was successful.
  6. Use **pd.get_dummies()** to encode categorical variables.
  7. Split the preprocessed data into a features array, **X**, and a target array, **y**. Use these arrays and the **train_test_split** function to split the data into training and testing datasets.
  8. Scale the training and testing features datasets by creating a **StandardScaler** instance, fitting it to the training data, then using the **transform** function.
 
### 2. Compile, Trian, and Evaluate the Model
Using your knowledge of TensorFlow, you'll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You'll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you've completed that step, you'll compile, train, and evaluate your binary classification model to calculate the model's loss and accuracy.
1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3. Create the first hidden layer and choose an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropirate activation function.
6. Check the structure of the model
7. Compile and train the model
8. Create a callback that saves the model's weights every five epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file. Name the file **AlphabetSoupCharity.h5**

### 3. Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.
Use any or all of the following methods to optimize your model:
 * Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
   * Dropping more or fewer columns.
   * Creating more bins for rare occurrences in columns.
   * Increasing or decreasing the number of values for each bin.
   * Add more neurons to a hidden layer.
   * Add more hidden layers.
   * Use different activation functions for the hidden layers.
   * Add or reduce the number of epochs to the training regimen.
- **Note:** If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.
1. Create a new Google Colab file and name it **AlphabetSoupCharity_Optimization.ipynb**
2. Import your dependencies and read in the **charity_data.csv** to a Pandas DataFrame.
3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
5. Save and export your results to an HDF5 file. Name the file **AlphabetSoupCharity_Optimization.h5**.

### 4. Write a Report on the Neural Network Model
For this part of the assignment, you'll write a report on the performance of the deep learning model you created for Alphabet Soup.
- The report should contain the following:
1. **Overview** of the analysis: Explain the purpose of this analysis.
2. **Results**: Using bulleted lists and images to support your answers, address the following questions:
      * Data Preprocessing:
        * What variable(s) are the target(s) for your model?
        * What variable(s) are the features for your model?
        * What variable(s) should be removed from the input data because they are neither targets nor features?
      * Compiling, Training, and Evaluating the Model
        * How many neurons, layers, and activation functions did you select for your neural network model, and why?
        * Were you able to achieve the target model performance?
        * What steps did you take in your attempts to increase model performance?
3. **Summary**: Summarize the overall results of the deep leanring model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
     
### 5. Copy Files Into Your Repository
Now that you're finished with your analysis in Google Colab, you nee to get your files into your repository for final submission.
1. Download your Colab notebooks to your computer
2. Move them into your Deep Learing Challenge directory in your local repository.
3. Push the added files to GitHub

------------------------------------
References
------------------------------------
Everything included in this project was covered in class. Regardless, these two resources were used to complete the challenge.
- Using Keras for Machine Learning:
  - https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/

- Learning Rate Scheduler:
  - https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
  - https://keras.io/api/callbacks/learning_rate_scheduler/
  - https://d2l.ai/chapter_optimization/lr-scheduler.html
  - https://stackoverflow.com/questions/61981929/how-to-change-the-learning-rate-based-on-the-previous-epoch-accuracy-using-keras
  - https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler

- Validation_Split function:
  - https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
  - https://datascience.stackexchange.com/questions/38955/how-does-the-validation-split-parameter-of-keras-fit-function-work

- Activation Functions:
  - https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/

- Optimizers:
  - https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/
  - https://musstafa0804.medium.com/optimizers-in-deep-learning-7bf81fed78a0
  - https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6

- Callbacks:
  - https://www.kdnuggets.com/2019/08/keras-callbacks-explained-three-minutes.html
  - https://medium.com/@ompramod9921/callbacks-your-secret-weapon-in-machine-learning-b08ded5678f0
  - https://www.tensorflow.org/guide/keras/writing_your_own_callbacks

- Saving and Loading Models: 
  - https://colab.research.google.com/github/agungsantoso/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%206%20-%20Saving%20and%20Loading%20Models.ipynb
  - https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_load.ipynb
  - https://stackoverflow.com/questions/64808087/how-do-i-save-files-from-google-colab-to-google-drive
  - https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory
