import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load dataset from a hypothetical CSV file location
dataset = pd.read_csv(r'Bank_Customer_Churn_Prediction/Churn_Modelling.csv')

# Remove unnecessary columns
dataset.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# Encode categorical variables
labelencoder = LabelEncoder()
dataset['Gender'] = labelencoder.fit_transform(dataset['Gender'])
dataset['Geography'] = labelencoder.fit_transform(dataset['Geography'])

# Define features and target variable
X = dataset.drop(columns='Exited')  # Features
y = dataset['Exited']  # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=40)

# Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Model evaluation for Logistic Regression
y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("Logistic Regression Accuracy: {:.2f}%".format(accuracy_logistic * 100))
print(classification_report(y_test, y_pred_logistic))

# Plot ROC Curve for Logistic Regression
y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_prob_logistic)

# Gradient Boosting model
gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=40)
gradient_boosting_model.fit(X_train, y_train)

# Model evaluation for Gradient Boosting
y_pred_gradient_boosting = gradient_boosting_model.predict(X_test)
accuracy_gradient_boosting = accuracy_score(y_test, y_pred_gradient_boosting)
print("Gradient Boosting Accuracy: {:.2f}%".format(accuracy_gradient_boosting * 100))
print(classification_report(y_test, y_pred_gradient_boosting))

# Plot ROC Curve for Gradient Boosting
y_prob_gradient_boosting = gradient_boosting_model.predict_proba(X_test)[:, 1]
fpr_gradient_boosting, tpr_gradient_boosting, _ = roc_curve(y_test, y_prob_gradient_boosting)

# Plot ROC Curves
plt.figure(figsize=(10, 5))
plt.plot(fpr_logistic, tpr_logistic, color='blue', lw=2, label='Logistic Regression')
plt.plot(fpr_gradient_boosting, tpr_gradient_boosting, color='red', lw=2, label='Gradient Boosting')
plt.plot([0, 1], [0, 1], color='brown', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()
