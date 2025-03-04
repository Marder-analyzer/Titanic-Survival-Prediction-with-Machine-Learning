# Titanic Machine Learning Project

This project aims to predict the survival probability of passengers in the Titanic disaster using machine learning techniques. The study covers data analysis, visualization, and modeling in detail.

## 1. Project Description

In this study, the Titanic dataset is used to predict passenger survival based on demographic and ticket information. The dataset is sourced from Kaggle's "Titanic - Machine Learning from Disaster" competition.

## 2. Libraries Used

This project utilizes the following Python libraries:

- `pandas` - Data processing and manipulation
- `numpy` - Numerical computations
- `matplotlib` & `seaborn` - Data visualization
- `sklearn` - Machine learning models and data preprocessing

## 3. Dataset and Features

The dataset includes the following features:

- `PassengerId` - Passenger ID
- `Survived` - Survival status (0 = No, 1 = Yes)
- `Pclass` - Ticket class (1, 2, 3)
- `Name` - Passenger name
- `Sex` - Gender
- `Age` - Age
- `SibSp` - Number of siblings/spouses aboard
- `Parch` - Number of parents/children aboard
- `Ticket` - Ticket number
- `Fare` - Ticket fare
- `Cabin` - Cabin number
- `Embarked` - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## 4. Data Preprocessing

The following preprocessing steps were applied:

- Handling missing values (e.g., missing `Age` values were filled with the mean)
- Converting categorical variables into numerical representations (`Sex` and `Embarked` were encoded using `Label Encoding`)
- Dropping irrelevant columns (`Name`, `Ticket`, `Cabin` were removed due to low predictive impact)
- Scaling numerical values (`Fare` and `Age` were normalized using `StandardScaler`)

## 5. Data Analysis and Visualization

The following analyses and visualizations were performed:

- Survival rates based on age, gender, and ticket class using `seaborn` and `matplotlib`
- Correlation analysis to determine key features
- Distribution analysis for categorical variables

## 6. Machine Learning Models

Several machine learning algorithms were applied for survival prediction:

- **Logistic Regression**
- **Decision Trees (Decision Tree Classifier)**
- **Random Forest (Random Forest Classifier)**
- **Support Vector Machines (SVM)**
- **Neural Networks (MLPClassifier)**

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve

## 7. Model Performance Comparison

At the end of the study, the best-performing model was selected based on accuracy and other evaluation metrics. The **Random Forest Classifier** was identified as the most effective model for this dataset.

## 8. Conclusion and Insights

Through the analysis and modeling of the Titanic dataset, passenger survival probabilities were predicted with high accuracy. Key findings include:

- Gender, age, and ticket class significantly influenced survival rates.
- Female passengers and those in first class had a higher likelihood of survival.
- Certain features had minimal impact on predictions and were removed for model efficiency.

This project serves as an excellent starting point for exploring basic machine learning techniques and understanding model evaluation.

## 9. How to Run

To run this project, follow these steps:

1. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
2. Execute the Python script:
   ```bash
   python titanic_ml.py
   ```

## 10. References

- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- Various resources on machine learning and data analysis

## 11. Recommendation for Beginners

The Titanic dataset is an excellent resource for beginners in machine learning. It includes both categorical and numerical variables, making it ideal for practicing data analysis, preprocessing, visualization, and model training.

Additionally, it provides a great opportunity to learn how to handle missing data, transform categorical variables, and compare different machine learning models. 

If you are new to machine learning, this dataset is a great way to develop essential skills. Feel free to analyze the code, modify it according to your needs, and experiment with different models to improve your understanding. By exploring and testing different approaches, you can gain hands-on experience and build confidence in data science and machine learning!
