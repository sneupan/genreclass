import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Handle missing values
    data.dropna(inplace=True)
    data = data.drop(['track_id', 'artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit'], axis=1)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['key'] = label_encoder.fit_transform(data['key'])
    data['mode'] = label_encoder.fit_transform(data['mode'])
    
    return data

def perform_eda(data):
    # Analyze genre distribution
    genre_counts = data['track_genre'].value_counts()
    plt.figure(figsize=(12, 8))
    genre_counts.plot(kind='bar')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.title('Genre Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    
    # Select only numeric columns for correlation analysis
    numeric_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    data_numeric = data[numeric_columns]
    
    # Analyze feature correlations
    corr_matrix = data_numeric.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()

def select_features(X, y):
    # Perform feature selection using SelectKBest
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_feature_names = X.columns[selector.get_support()]
    
    return X_selected, selected_feature_names


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    for name, model in models.items():
        if name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100],  # Reduced number of estimators
                'max_depth': [None, 10],  # Reduced number of max_depth values
                'min_samples_split': [2, 5]  # Reduced number of min_samples_split values
            }
        else:
            param_grid = {}
        
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} - Best Parameters: {grid_search.best_params_}")
        print(f"{name} - Accuracy: {accuracy}")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{name} - Confusion Matrix')
        plt.show()
        
        print()

def main():
    file_path = 'dataset.csv'
    data = load_data(file_path)
    data = preprocess_data(data)
    
    perform_eda(data)
    
    X = data.drop('track_genre', axis=1)
    y = data['track_genre']
    
    # Perform feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform feature selection
    X_selected, selected_features = select_features(pd.DataFrame(X_scaled, columns=X.columns), y)
    print("Selected Features:", selected_features)
    
    # Split the data into train and test sets (using a smaller subset)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, train_size=0.1, random_state=42)
    
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()