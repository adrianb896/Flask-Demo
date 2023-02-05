from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def roc2():
    # Importing dataset
    df = pd.read_csv("cc.csv")

    # Splitting the data into features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Training the model
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    # Predicting the test set results
    y_pred_proba = classifier.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)

    # Plotting the ROC curve
    plt.plot(fpr,tpr)
    # Plot the random guessing line
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guessing')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
