import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


pd.options.display.max_columns = None
pd.options.display.max_rows = None


# Improving dataset
file_path = r'C:\Users\yayua\Desktop\SPU\Summer2023\ISM6358\preprocessed_data.xlsx'
data = pd.read_excel(file_path)
data['Credit Risk'] = data['Credit Risk'].replace({'High': 1, 'Low': 0})
print(f"Data:\n{data.head}")


# One-Hot Encoding
categorical_cols = ['Loan Purpose', 'Gender', 'Marital Status', 'Housing', 'Job']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)


# Modeling
# Identify X and Y
X = data_encoded
y = data_encoded.pop('Credit Risk')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)


# Finding the model with highest acc
models = []
models.append(('RF', RandomForestClassifier()))
models.append(('LinearSVM', SVC(kernel='linear')))
models.append(('NonLinearSVM', SVC(kernel='rbf')))
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))


folds = 10
kfold = KFold(n_splits=folds, random_state=42, shuffle=True)
cv_df = pd.DataFrame(index=range(folds * len(models)))


entries = []
for model_name, model in models:
    # model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=kfold)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print(cv_df)


plt.figure(figsize=[15, 10])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
print(plt.show())
print(cv_df.groupby('model_name').accuracy.mean().sort_values(ascending=False))


# Create a logistic regression model
logreg = LogisticRegression()

# Fit the model on the training data
logreg.fit(X_train, y_train)

# Predict on the training set
y_train_pred = logreg.predict(X_train)
print("----------Train Set - classification report ----------\n")
print(classification_report(y_train, y_train_pred))

# Predict on the test set
y_test_pred = logreg.predict(X_test)
print("----------Test Set - classification report ----------\n")
print(classification_report(y_test, y_test_pred))

# Plot ROC curve
plt.title("ROC Curve")
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='thistle', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()

# Print ROC value
logreg_roc_value = roc_auc_score(y_test, y_test_pred)
print("Logistic Regression ROC value: {0}".format(logreg_roc_value))

# Print test accuracy and confusion matrix
test_accuracy = round(accuracy_score(y_test, y_test_pred), 2) * 100
print("Test Accuracy is: ", test_accuracy)
print(confusion_matrix(y_test, y_test_pred))


# Predicting Additional Data based on the existing dataset
# Load the additional data from Excel file
additional_data = pd.read_excel('C:/Users/yayua/Desktop/SPU/Summer2023/ISM6358/Additional_data.xlsx')
additional_data['Total Balance'] = additional_data['Checking '] + additional_data['Savings']

print("======================================================================\n"
      "Fitting the model for additional data\n"
      "**********************************************************************\n"
      "Making sure they are in same page...\n")

# Compare column names
data_cols = set(data.columns)
additional_data_cols = set(additional_data.columns)

if data_cols == additional_data_cols:
    print("Column names are the same.")
else:
    print("Columns names are different.")


# Compare data types
data_dtypes = data.dtypes
additional_data_dtypes = additional_data.dtypes

different_dtypes = []
for col in data_cols:
    if data_dtypes[col] != additional_data_dtypes[col]:
        different_dtypes.append(col)

if len(different_dtypes) == 0:
    print("Data types are the same.")
else:
    print("Different data types found:")
    for col in different_dtypes:
        print(f"Column: {col} - data type in 'data': {data_dtypes[col]}, data type in 'additional_data': {additional_data_dtypes[col]}")


# Convert data types in additional_data to match data
additional_data['Months Employed'] = additional_data['Months Employed'].astype(float)
additional_data['Age'] = additional_data['Age'].astype(float)
additional_data['Months Customer'] = additional_data['Months Customer'].astype(float)
additional_data = additional_data.drop(columns=['Credit Risk'])
print(additional_data)


# Preprocess the additional data (perform the same preprocessing as done for training data)
add_categorical_cols = ['Loan Purpose', 'Gender', 'Marital Status', 'Housing', 'Job']
print(add_categorical_cols)

# Perform binary encoding for the categorical columns
add_data_encoded = pd.get_dummies(additional_data, columns=add_categorical_cols, drop_first=True)
print(add_data_encoded)


print('======================================================================\n'
      'Predicting Additional dataset')

# Assuming logreg is the trained logistic regression model
add_data_encoded = add_data_encoded.reindex(columns=data_encoded.columns, fill_value=0)
additional_data['Credit Risk'] = logreg.predict(add_data_encoded)


# Replace values in "Credit Risk" column
additional_data['Credit Risk'] = additional_data['Credit Risk'].replace({1: 'High', 0: 'Low'})


# Print the predicted credit risk for additional_data
print("Predicted Credit Risk for Additional Data:")
print(additional_data)


# Save additional_data with predicted credit risk as an Excel file
additional_data.to_excel('C:/Users/yayua/Desktop/SPU/Summer2023/ISM6358/Prediction.xlsx', index=False)

