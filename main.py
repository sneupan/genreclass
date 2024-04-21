
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def fetchData():
    file_path = 'dataset.csv'
    data = pd.read_csv(file_path)   
    return data

def generateTrainingData(data):
    features_columns = data.columns[8:20]  # From 'danceability' to 'time_signature'
    target_column = 'track_genre'
    selected_data = data.loc[:, list(features_columns) + [target_column]]


    label_encoder = LabelEncoder()
    selected_data[target_column] = label_encoder.fit_transform(selected_data[target_column])

    # Splitting the data into features (X) and target (y)
    X = selected_data.iloc[:, :-1].values
    y = selected_data.iloc[:, -1].values

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def trainingLogisticModel(X_train, X_test, y_train, y_test):

    # Create a logistic regression model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    # Train the model
    log_reg.fit(X_train, y_train)
    # Predict on the test set
    y_pred = log_reg.predict(X_test)
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def trainingRandomForests(X_train, X_test, y_train, y_test):
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on the training data
    random_forest.fit(X_train, y_train)
    # Predict on the test set
    y_pred_rf = random_forest.predict(X_test)
    # Calculate the accuracy
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    return accuracy_rf

def main():
    data = fetchData()
    X_train, X_test, y_train, y_test = generateTrainingData(data)
    print('Training...')
    accuracy = trainingRandomForests(X_train, X_test, y_train, y_test)

    print(accuracy)
    
main()