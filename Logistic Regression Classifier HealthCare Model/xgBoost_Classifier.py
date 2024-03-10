import numpy as np                                      # For matrix operations and numerical processing
import xgboost as xgb                                   # For XGBoost model training and evaluation
from sklearn.model_selection import train_test_split    # For splitting the dataset
from sklearn.metrics import accuracy_score              # For evaluating the model

# Logistic Regression Classifier Model:
# A Highly efficient and scalable implementation
# of gradient boosting machine learning technique

def load_data():
    """
    Loads and returns a hypothetical healthcare dataset
    (DATASET TO TRAIN THE MODEL)

    Returns:
        X (numpy array): Health metrics matrix
        y (numpy array): Target vector indicating diabetes presence (0 or 1)
    """
    # Hypothetical data: [Blood Pressure, BMI, Age, Glucose Level]
    # X = np.array([
    #     [120, 24.5, 45, 85],
    #     [130, 30.2, 50, 120],
    #     [115, 22.0, 36, 80],
    #     [140, 28.6, 52, 180]])
    X = np.array([
        [120, 24.5, 45, 85],
        [130, 30.2, 50, 120],
        [115, 22.0, 36, 80],
        [140, 28.6, 52, 180],
        [125, 26.0, 43, 95],
        [135, 31.1, 55, 125],
        [110, 21.5, 37, 82],
        [145, 29.2, 53, 175],
        [128, 27.3, 48, 105],
        [138, 32.8, 58, 130],
        [118, 23.4, 40, 88],
        [142, 30.9, 54, 185],
        [123, 25.7, 46, 92],
        [136, 33.5, 60, 135],
        [113, 20.2, 35, 78],
        [148, 28.0, 51, 170],
        [126, 24.8, 44, 90],
        [139, 29.6, 56, 128],
        [117, 22.9, 39, 85],
        [143, 31.4, 57, 182]
    ])

    # Corresponding diabetes presence outcomes: 1 for presence, 0 for absence
    # y = np.array([0, 1, 0, 1])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Returns the expanded dataset
    return X, y

def split_data(X, y):
    """
    Splits the dataset into training and testing sets
    (SEPARATE THE DATASET INTO TRAINING AND TESTING SETS)
    
    Args:
        X (numpy array): Health metrics matrix
        y (numpy array): Target vector for diabetes presence
    
    Returns:
        Split dataset (training and testing sets)
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """
    Trains the XGBoost model for diabetes risk prediction
    (A TRAINED MODEL TO PREDICT)
    
    Args:
        X_train (numpy array): Training health metrics
        y_train (numpy array): Training target for diabetes presence

    Returns:
        Trained XGBoost model
    """
    # Predicts the probability of diabetes presence for binary classification
    model = xgb.XGBClassifier(objective='binary:logistic')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance on the test set
    (PRODUCES THE ACCURACY OF THE MODEL - PERCENTAGE OF CORRECT PREDICTIONS)
    
    Args:
        model: Trained XGBoost model
        X_test (numpy array): Testing health metrics
        y_test (numpy array): Testing target for diabetes presence
    
    Returns:
        Accuracy of the model
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main():
    """
    Main function for the diabetes risk prediction process
    """
    # Load and split data
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)

    # Print the accuracy
    print(f"Model Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
