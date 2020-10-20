import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

col_names = ['Name','Calories','Category','When','Plan','veg','clusterID','Meal','Fat','Cholesterol','sodium','Potassium','Carbohydrates','sugars','iron','Lable']
food = pd.read_csv('fooddataset.csv', header=None, names=col_names)
print(food.head())



feature_cols = ['Carbohydrates', 'sugars']
X = food[feature_cols] # Features
y = food.Lable # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
