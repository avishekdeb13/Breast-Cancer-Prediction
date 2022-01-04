# Libraries to import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Datafile load
datafile = pd.read_csv("E:/PROJECTS/Machine Learning/4. Breast cancer detection/Dataset/breastCancer.csv")
df = datafile.copy()

df.info
df.dtypes
df.isnull().sum()

## Lets study the individual columns

# clump_thickness
df.clump_thickness.value_counts()
sns.countplot("clump_thickness", data=df)
plt.title("Clump thickness distribution")

sns.countplot("clump_thickness", hue="class", data=df)
plt.title("Clump thickness distribution across classes")

# size_uniformity
df.size_uniformity.value_counts()

# shape_uniformity
df.shape_uniformity.value_counts()

# bare_nucleoli
df.bare_nucleoli.value_counts()
df.bare_nucleoli.unique_values()

df[df['bare_nucleoli'] == '?'].sum()
# There are special characters (?) in this column, we can reload the dataset excluding these characters
# or we can imputate them with some values.
# we will impute them with the central tendency

df = df.replace('?', np.nan)

df.median()
df.mean()

df = df.fillna(df.median())

df.bare_nucleoli.dtypes
df.dtypes

# Only the bare_nucleoli column has object as the datatype, lets change it to int
df['bare_nucleoli'] = df['bare_nucleoli'].astype('int64')

# Drpping the id colum since it's of no use to us
df.drop('id', axis=1, inplace=True)

df.describe().T

# All the variable distribution
df.hist(bins=20, figsize=(30, 30), layout=(6, 3))
plt.title("Variable distribution")

# Distribution of the categorical variable
sns.distplot(df['class'])
plt.title("Distribution of the categorical variable")

# Checking the presence of outliers in the datatset
plt.figure(figsize=(15, 10))
sns.boxplot(data=df, orient="h")
plt.title("5 point summary visualisation")

# Checking the correlation between all the variables
plt.figure(figsize=(35, 15))
sns.heatmap(df.corr(), vmax=1, square=True, annot=True, cmap='viridis')
plt.title('Correlation between different attributes')
plt.show()
# Thus we can see that the variables are not much related to each other other than
# Shape_uniformity and size uniformity has high correlation among themselves.

# Separation of the dependent and independent variable
x = df.drop('class', axis=1)
y = df['class']

# To check for multicollinearity in the datatset
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns

vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                   for i in range(len(x.columns))]
print(vif_data)

# Shape_uniformity and size_uniformity has high vif, thus we need to scale the data to reduce the vif
# Scaling
scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)

# VIF check after scaling
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
vif["Features"] = x.columns
vif

# We can see the vif has reduced.


# Train test split of the datatset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)

## Model Building

# LOGISTIC REGRESSION
model_log = LogisticRegression()
model_log.fit(x_train, y_train)
a = model_log.score(x_train, y_train)
print("                 LOGISTIC MODEL           ")
print("                                          ")
print(f"Model accuracy on train dataset : {round(a, 2)}")

y_pred = model_log.predict(x_test)
b = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test data:      {round(b, 2)}")
print("---------" * 10)

c = confusion_matrix(y_test, y_pred)
print(f"The confusion martix is: \n {c}")
print("---------" * 10)

d = classification_report(y_test, y_pred)
print(f"The classification report of the model: \n {d}")

# Logistic regression gives us accuracy of 96% with a recall rate of 96%
# In this use case, we must aim for high recall rate as the model should be able to recall as maximum
# number of patients with breast cancer

## SUPPORT VECTOR CLASSIFIER
model_svc = SVC()
model_svc.fit(x_train, y_train)
a = model_svc.score(x_train, y_train)
print("                 SUPPORT VECTOR CALSSIFIER MODEL           ")
print("                                          ")
print(f"Model accuracy on train dataset : {round(a, 2)}")

y_pred = model_svc.predict(x_test)
b = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test data:      {round(b, 2)}")
print("---------" * 10)

c = confusion_matrix(y_test, y_pred)
print(f"The confusion martix is: \n {c}")
print("---------" * 10)

d = classification_report(y_test, y_pred)
print(f"The classification report of the model: \n {d}")

# Better model performance of 97% accuracy with 97% recall rate

##Lets try SVC with little parameters change to see if the accuracy improves
model_svc1 = SVC(gamma=0.25, C=3)
model_svc1.fit(x_train, y_train)
a = model_svc1.score(x_train, y_train)
print("                 SUPPORT VECTOR CALSSIFIER MODEL WITH PARAMETER TUNING   ")
print("                                          ")
print(f"Model accuracy on train dataset : {round(a, 2)}")

y_pred = model_svc1.predict(x_test)
b = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test data:      {round(b, 2)}")
print("---------" * 10)

c = confusion_matrix(y_test, y_pred)
print(f"The confusion martix is: \n {c}")
print("---------" * 10)

d = classification_report(y_test, y_pred)
print(f"The classification report of the model: \n {d}")

## This model has over-fitted after parameter tuning as we can see the test data score accuracy reducing whereas
# the model has achieved 100% accuracy on the train dataset

##KNN CLASSIFIER
model_knn = KNeighborsClassifier()
model_knn.fit(x_train, y_train)
a = model_knn.score(x_train, y_train)
print("                      KNN MODEL           ")
print("                                          ")
print(f"Model accuracy on train dataset : {round(a, 2)}")

y_pred = model_knn.predict(x_test)
b = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test data:      {round(b, 2)}")
print("---------" * 10)

c = confusion_matrix(y_test, y_pred)
print(f"The confusion martix is: \n {c}")
print("---------" * 10)

d = classification_report(y_test, y_pred)
print(f"The classification report of the model: \n {d}")

# The model gives us almost the same result as the SVC model, thus we can chose any of the model as our final
# model for prediction

# Saving the best  model
pickle.dump(model_knn, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
