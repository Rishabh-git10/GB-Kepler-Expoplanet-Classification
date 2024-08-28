# Kepler Exoplanet Classification using Gradient Boosting

## Overview
This project aims to classify exoplanets using data from the Kepler telescope. The classification is performed using a Gradient Boosting Classifier, which is a powerful ensemble learning method. The project involves data cleaning, preprocessing, feature engineering, and hyperparameter tuning to optimize the model's performance.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
The Kepler telescope has provided a wealth of data on stellar characteristics, which can be used to predict the existence of exoplanets. This project leverages Gradient Boosting, a robust machine learning technique, to classify exoplanets based on these characteristics.

## Dataset
The dataset used in this project is obtained from the Kepler telescope. It contains various stellar characteristics and a target variable indicating the presence of an exoplanet.

## Installation
To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- scipy

You can install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn matplotlib scipy
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/rishabh-git10/gb-kepler-exoplanet-classification
   ```
2. Navigate to the project directory:
   ```bash
   cd gb-kepler-exoplanet-classification
   ```
3. Run the main notebook:
   ```bash
   python main.ipynb
   ```

## Project Structure
- `main.ipynb`: The main notebook that runs the entire pipeline.
- `dataset.csv`: The dataset file containing the Kepler telescope data.
- `README.md`: This readme file.

## Data Preprocessing
The data preprocessing steps include:
1. **Handling Missing Values**: Missing values are imputed using the mean strategy.
2. **Encoding Categorical Features**: Categorical features are encoded using Label Encoding.
3. **Scaling Features**: Features are scaled using StandardScaler to normalize the data.

## Feature Engineering
Feature engineering involves creating new features that may be relevant to the classification task. In this project, we focus on selecting the most important features based on their importance scores from the Gradient Boosting model.

## Model Training and Evaluation
The model training and evaluation process includes:
1. **Hyperparameter Tuning**: Using RandomizedSearchCV to find the best hyperparameters for the Gradient Boosting Classifier.
2. **Cross-Validation**: Performing cross-validation to evaluate the model's performance.
3. **Final Evaluation**: Evaluating the model on the test set to determine its accuracy.

## Results
The Gradient Boosting Classifier achieved a test accuracy of 93.8%. The model's performance was optimized through extensive hyperparameter tuning and feature selection.

## Visualization
The project includes visualizations to understand the model's performance and the importance of different features:
- Scatter plot of learning rate vs. number of estimators colored by mean test score.
- Scatter plot of true values vs. predictions to visualize the model's accuracy.

## Conclusion
This project demonstrates the effectiveness of Gradient Boosting in classifying exoplanets using data from the Kepler telescope. The model achieved a high accuracy, indicating its potential for real-world applications in astronomy.

## References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/index.html)

Feel free to explore the code and modify it to suit your needs. If you have any questions or suggestions, please open an issue or contact the author [@rishabh-git10](https://github.com/rishabh-git10).

---

This README provides a comprehensive guide to understanding and running the project. It includes all necessary steps and explanations to ensure that users can easily follow along and reproduce the results.