# Medical-insurance-cost-prediction.ipynb
his project aims to predict medical insurance costs using machine learning techniques. It leverages the Linear Regression algorithm to predict the charges based on various features such as age, sex, BMI, number of children, smoking status, and region.

The dataset for this project can be found at https://www.kaggle.com/datasets/mirichoi0218/insurance

## Table of Contents

- [Overview](#overview)
- [Dependencies](#Dependencies)
- [Dataset Information](#DatasetInformation)
- [Data Analysis](#DataAnalysis)
- [Data Pre-Processing](#Datapre-processing)
- [Model Training](#ModelTraining)
- [Model Evaluation](#ModelEvaluation)
- [Predictive System](#PredictiveSystem)
- [How to Run the Code](#Howtorunthecode)

 ## overview
In this project, we use a medical insurance dataset to train a Linear Regression model to predict insurance costs. This is a supervised learning problem where the target variable is the insurance charges (the amount to be paid for the medical insurance).

 ## The project includes the following steps
 - Data Collection
- Data Analysis
- Data Pre-processing (handling categorical data, encoding)
- Model Training and Evaluation using Linear Regression
- Building a Predictive System

 ## Dependencies
Ensure that you have the following dependencies installed to run this code.

- ### Python 3.x
- ### NumPy: For numerical computing
- ### Pandas: For data manipulation
- ### Matplotlib: For data visualization
- ### Seaborn: For statistical data visualization
- ### Scikit-learn: For machine learning and model evaluation
  
- Install dependencies using pip if not already installed:

pip install numpy pandas matplotlib seaborn scikit-learn

 ## Dataset Information
The dataset consists of the following columns:

- age: Age of the insured person.
- sex: Gender of the insured (male/female).
- bmi: Body Mass Index (BMI) of the insured.
- children: Number of children/dependents covered by the insurance.
- smoker: Whether the person is a smoker (yes/no).
- region: The region where the insured person resides (southeast, southwest, northeast, northwest).
- charges: The medical insurance charges for the individual (target variable).

  ## Data Analysis
Data analysis involves exploring the dataset and performing visualization on the different features:

- Distribution of age and BMI
- Gender distribution
- Children and smoker columns visualization
- Region distribution
- Charges distribution

- This helps in understanding the dataset, identifying outliers, missing values, and relationships between features.

  ##  Data Pre-Processing
 ### Categorical Features Encoding:
Since some of the features like sex, smoker, and region are categorical, we encode them into numeric values:

- sex: Male → 0, Female → 1
- smoker: Yes → 0, No → 1
- region: southeast → 0, southwest → 1, northeast → 2, northwest → 3
###  Splitting the Dataset:
- The dataset is split into independent features (X) and the target variable (Y).
- The dataset is further split into training (80%) and testing (20%) sets using train_test_split.

## Model Training
We train a Linear Regression model to predict the insurance charges. The model is trained using the training set and then evaluated on both the training and testing datasets to determine its accuracy and performance.

## Model Evaluation
After training the model, we evaluate its performance using the R-squared value (R²), which indicates how well the model fits the data. The R² value ranges from 0 to 1, with 1 indicating a perfect fit.

- Training Data R² Value: Measures how well the model performs on the training data.
- Test Data R² Value: Measures how well the model generalizes to unseen data.

 ## Predictive System
Once the model is trained and evaluated, we can use it to predict insurance charges for new individuals. Here's an example:

- Input: Age = 31, Sex = 1 (female), BMI = 25.74, Children = 0, Smoker = 1 (non-smoker), Region = 0 (southeast)
- Output: Predicted insurance charge for this individual.

   input_data =  (31, 1, 25.74, 0, 1, 0)
  
     input_data_as_numpy_array = np.asarray(input_data)

     input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
   
     prediction = regressor.predict(input_data_reshaped)

     print('The insurance cost is USD ', prediction[0])

## How to Run the Code
Download the dataset: https://www.kaggle.com/datasets/mirichoi0218/insurance

Load the dataset: Read the dataset using Pandas and perform exploratory data analysis.

Preprocess the data: Encode categorical variables and split the dataset into training and testing sets.

Train the model: Use the Linear Regression model from Scikit-learn to train on the dataset.

Evaluate the model: Check the performance on both training and testing sets.

Predict: Use the trained model to make predictions for new data.

## Conclusion
This project demonstrates how machine learning can be applied to predict medical insurance costs based on a person's characteristics and behavior. The linear regression model provides a simple yet effective method to estimate insurance charges, offering valuable insights into the factors influencing medical insurance costs.

For further questions or enhancements to the model, feel free to modify and adapt the code to suit your needs.








