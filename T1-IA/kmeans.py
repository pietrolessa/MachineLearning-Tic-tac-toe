from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import numpy as np

# Initialize the Gaussian Naive Bayes classifier
classifier = GaussianNB()

# Optionally, add a function to evaluate the classifier using cross-validation
def evaluate_classifier(x_train, y_train, cv=5):
    """
    Evaluates the GaussianNB classifier using cross-validation.
    
    Parameters:
    - x_train: Training data features
    - y_train: Training data labels
    - cv: Number of cross-validation folds (default is 5)
    
    Returns:
    - mean_accuracy: Mean accuracy over the cross-validation folds
    - std_accuracy: Standard deviation of accuracy over the cross-validation folds
    """
    scores = cross_val_score(classifier, x_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)
    
    print(f"Cross-validation mean accuracy: {mean_accuracy:.2f}")
    print(f"Cross-validation standard deviation: {std_accuracy:.2f}")
    
    return mean_accuracy, std_accuracy

# Example usage of the evaluate_classifier function
# x_train, y_train should be your training data
# mean_accuracy, std_accuracy = evaluate_classifier(x_train, y_train)
