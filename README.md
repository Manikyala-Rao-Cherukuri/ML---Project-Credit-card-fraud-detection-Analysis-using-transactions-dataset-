# Project Title: Fraud Detection with Random Forest Classifier
A machine learning project completed during a company hackathon competition.
#### Libraries Used
- numpy
- pandas
- pyplot
- seaborn
- sklearn.model_selection
- sklearn.ensemble
- sklearn.metrics
#### Data Source
The data file was obtained from Kaggle at the following link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#### Data Cleaning
Null values were removed from the dataset.
#### Data Balancing
An over-sampling approach was used to balance the imbalanced dataset.
#### Data Splitting
The dataset was split into training and testing sets using the train_test_split library from sklearn.model_selection.
#### Model Training
The RandomForestClassifier was imported from sklearn.ensemble.
The model was trained using n_estimators=641 decision trees.
#### Model Evaluation
The testing data was used to evaluate the trained model.
The number of errors was noted.
The confusion_matrix and accuracy_score were imported from sklearn.metrics.
The confusion matrix was drawn using y_pred and y_test data.
The model was found to have 99.99% efficiency.
#### DataFrame Creation
A Python function was created to create multiple columns in a DataFrame.
The x_test data was loaded into the DataFrame.
#### Result Comparison
The y_actual and y_predicted data were compared.
Colors were added to highlight differences between the actual and expected output.
The final output was exported to Excel in the current directory.
#### Conclusion
The Random Forest Classifier proved to be an effective method for detecting fraud in the dataset.
Future work could explore other methods for handling imbalanced data and further optimize the model.
