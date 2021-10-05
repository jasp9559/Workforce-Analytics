'''Project work for Workforce Analytics, wherein we need to work on the available features 
   to figure out a solution on the attrition and forecast the problem in order to help 
   stabilise the attrition rate and sturdy the features to help the businiess flow be proper'''

'''We start by importing the data and having a look at the various features gathered for the problem'''

# Importing the libraires
import pandas as pd
import numpy as np


attrition = pd.read_excel("C:/Data Science/Data Science/HR Workforce Analytics Project/Final dataset Attrition.xlsx")
attrition.head(10)
attrition.columns
attrition.shape 

''' The dataset gathered has 1,470 nos of observations and the following 32 nos of features
1. "Age"                        = The age of the employee
2. "Attrition"                  = Whether the employee has attrited or not
3. "BusinessTravel"             = Whether the employee used to travel for business or not
4. "Department"                 = Which department the employee was employed under
5. "DistanceFromHome"           = The distance the employee travels to reach for job on a day to day basis
6. "Gender"                     = Gender of the employee
7. "JobInvolvement"             = The involvement rating of an employee over the job handled
8. "JobLevel"                   = Level at which the employee is working
9. "JobRole"                    = The roles and resposibilites of the employee
10. "JobSatisfaction"           = Satisfaction rating of the employee for the job
11. "MaritalStatus"             = Marital status of the employee
12. "MonthlyIncome"             = Monthly income of the employees
13. "NumCompaniesWorked"        = Number of companies the employees has worked for
14. "OverTime"                  = Whether working Overtime or not
15. "PercentSalaryHike"         = Percentage salary hike since their appointment in the company
16. "PerformanceRating"         = Performance rating 
17. "StockOptionLevel"          = Level of opted for sharing the stock 
18. "TotalWorkingYears"         = Total years worked by the employees
19. "TrainingTimesLastYear"     = How many trainings the employee has undergone
20. "YearsAtCompany"            = Years spent at the present organisation
21. "YearsSinceLastPromotion"   = Time gone in years since last promotion
22. "YearsWithCurrManager"      = Years working under he current manager
23. "Higher_Education"          = Higher education level of the employee
24. "Date_of_Hire"              = Date of hire of the employee in the current organisation
25. "Date_of_termination"       = Date of termination from the organisation
26. "Status_of_leaving"         = Reason for leaving the organisation
27. "Mode_of_work"              = WFH or WFO
28. "Leaves"                    = Total permitted leaves taken by the employee
29. "Absenteeism"               = Total days absent for the employee
30. "Work_accident"             = Work accident if any
31. "Source_of_hire"            = Source of hire
32. "Job_Mode"                  = Working full time/ part or contractual
'''

attrition.describe() 

# Checking whether the datset has any missing values within
attrition.isna().sum() 



# Category columns in the data
category_cols = ['Attrition', 'BusinessTravel', 'Department', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'Higher_Education', 'Status_of_leaving', 'Mode_of_work', 'Work_accident', 'Source_of_Hire', 'Job_mode']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
attrition[category_cols] = attrition[category_cols].apply(le.fit_transform)
attrition 

# removing/ dropping the columns passenger id, Name, ticket, cabin
attrition = attrition.drop(["Date_of_Hire", "Date_of_termination"], axis = 1)
attrition

'''
The data seems to be normal and has values that are not irrelevant
Additionally we have to now normalize the data as the scale is not the same
 for all the variables. We will use minmax scaler for the job

'''

from sklearn.preprocessing import MinMaxScaler as mms
scale = mms() 
attrition_mms = pd.DataFrame(scale.fit_transform(attrition.iloc[:,:])) 
attrition_mms.columns = attrition.columns

attrition_mms.corr() 
corr_matrix = attrition_mms.corr() 
(corr_matrix['Attrition'].sort_values(ascending = False)) 
'''We notice the correlation of various features and find that 
   OverTime and Marital Status has the highest corelation with the Attririon'''

'''EDA = Performing general EDA over the data'''

EDA = {"column": attrition_mms.columns,
       "mean": attrition_mms.mean(),
       "median": attrition_mms.median(),
       "mode": attrition_mms.mode(),
       "standard deviation": attrition_mms.std(),
       "variance": attrition_mms.var(),
       "skewness": attrition_mms.skew(),
       "kurtosis": attrition_mms.kurt()}

print(EDA)

# Lets check out some visualisation to get the insights on the data
df_company = attrition_mms 

import seaborn as sns 
import matplotlib.pyplot as plt
def stacked_plot(df, group, target):
    """
    Function to generate a stacked plots between two variables
    """
    fig, ax = plt.subplots(figsize = (6,4))
    temp_df = (df.groupby([group, target]).size()/df.groupby(group)[target].count()).reset_index().pivot(columns=target, index=group, values=0)
    temp_df.plot(kind = 'bar', stacked = True, ax = ax, color = ["green", "darkred"])
    ax.xaxis.set_tick_params(rotation=0)
    ax.set_xlabel(group)
    ax.set_ylabel('Attrition')

def Age(a):
    if a <= 25:
        return 1
    elif a > 25 and a <= 32:
        return 2
    elif a > 32 and a <= 40:
        return 3
    elif a > 40 and a <= 50:
        return 4
    else:
        return 5

df_company["Age_group"] = df_company["Age"].apply(lambda x: Age(x))
df_company["Age_group"].value_counts()
sns.countplot(x = "Age_group", hue = "Attrition", data = df_company)

def DistanceFromHome(d):
    if d <= 5:
        return 1
    elif d > 5 and d <= 10:
        return 2
    elif d > 10 and d <= 15:
        return 3
    elif d > 15 and d <= 20:
        return 4
    elif d > 20 and d <= 25:
        return 5
    else:
        return 6
    
df_company["DistanceFromHome_group"]=df_company["DistanceFromHome"].apply(lambda x: DistanceFromHome(x))
df_company["DistanceFromHome_group"].value_counts()
sns.countplot(x="DistanceFromHome_group", hue="Attrition", data=df_company)

def YearsAtCompany(t):
    if t <= 1:
        return 1
    elif t > 1 and t <= 5:
        return 2
    elif t > 5 and t <= 10:
        return 3
    elif t > 10 and t <= 20:
        return 4
    elif t > 20 and t <= 30:
        return 5
    else:
        return 6

df_company["YearsAtCompany"] = df_company["YearsAtCompany"].apply(lambda x:YearsAtCompany(x))
df_company["YearsAtCompany"].value_counts()
sns.countplot(x = "YearsAtCompany", hue = "Attrition", data = df_company)


stacked_plot(df_company, "Gender", "Attrition")
stacked_plot(df_company, "MaritalStatus", "Attrition")
stacked_plot(df_company, "BusinessTravel", "Attrition")
stacked_plot(df_company, "Department", "Attrition")
stacked_plot(df_company, "JobInvolvement", "Attrition")
stacked_plot(df_company, "JobRole", "Attrition")
stacked_plot(df_company, "JobLevel", "Attrition")
stacked_plot(df_company, "JobSatisfaction", "Attrition")
stacked_plot(df_company, "NumCompaniesWorked", "Attrition")
stacked_plot(df_company, "OverTime", "Attrition")
stacked_plot(df_company, "PercentSalaryHike", "Attrition")
stacked_plot(df_company, "PerformanceRating", "Attrition")
stacked_plot(df_company, "StockOptionLevel", "Attrition")
stacked_plot(df_company, "TrainingTimesLastYear", "Attrition")
stacked_plot(df_company, "Higher Education", "Attrition")
stacked_plot(df_company, "Status of leaving", "Attrition")
stacked_plot(df_company, "mode of work", "Attrition")
stacked_plot(df_company, "leaves", "Attrition")
stacked_plot(df_company, "absenteeism", "Attrition")
stacked_plot(df_company, "Work_accident", "Attrition")
stacked_plot(df_company, "Source of Hire", "Attrition")
stacked_plot(df_company, "Job mode", "Attrition")



















import matplotlib.pyplot as plt
import seaborn as sns
# we plot the pair plot to see the various conjunctions of the scater plots
sns.pairplot(attrition1.iloc[:,:], hue = "Attrition") 


plt.figure(figsize = (10,8))
sns.heatmap(attrition.corr(), annot = False, cmap = 'coolwarm')
plt.show()

corr_attr = attrition.corr()
(corr_attr['Attrition'].sort_values(ascending = False))

col = attrition.corr().nlargest(20, "Attrition").Attrition.index
plt.figure(figsize=(15, 15))
sns.heatmap(attrition[col].corr(), annot = True, cmap="RdYlGn", annot_kws={"size":10})




















# Splitting the data into train and test
from sklearn.model_selection import train_test_split

X = attrition.drop('Attrition', axis = 1)
y = attrition.Attrition

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train = True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict = True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

'''1. Decision Tree Classifier'''
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)


'''2. Decision Tree Classifier Hyperparameter tuning'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params = {
    "criterion":("gini", "entropy"), 
    "splitter":("best", "random"), 
    "max_depth":(list(range(1, 20))), 
    "min_samples_split":[2, 3, 4], 
    "min_samples_leaf":list(range(1, 20)), 
}

tree_clf = DecisionTreeClassifier(random_state=42)
tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)
tree_cv.fit(X_train, y_train)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params})")

tree_clf = DecisionTreeClassifier(**best_params)
tree_clf.fit(X_train, y_train)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

























'''
We start building the models for classification
We start by splitting the data into Train and test
'''
from sklearn.model_selection import train_test_split
X = attrition.iloc[:, 1:8]
Y = attrition.iloc[:, 0]

# herein we split the data with test size kept as 15%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state = 40)
print(y_train.value_counts())
print(y_test.value_counts())

# We start building the models using the following regression models for classifying
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

'''Logistic Regression'''
log = LogisticRegression() 
log.fit(x_train, y_train) 

log_acc = accuracy_score(y_test, log.predict(x_test)) 
print("Train Set Accuracy:"+str(accuracy_score(y_train, log.predict(x_train))*100)) 
print("Test Set Accuracy:"+str(accuracy_score(y_test, log.predict(x_test))*100)) 

plt.figure(figsize = (6,4)) 
df_ = pd.DataFrame(confusion_matrix(y_test, log.predict(x_test)), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_, annot=True,annot_kws={"size": 16}, fmt='g')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

'''Descision Tree'''
dec = DecisionTreeClassifier()
dec.fit(x_train, y_train)

dec_acc = accuracy_score(y_test, dec.predict(x_test))
print("Train test Accuracy:"+str(accuracy_score(y_train, dec.predict(x_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test, dec.predict(x_test))*100))

plt.figure(figsize = (6,4)) 
df_ = pd.DataFrame(confusion_matrix(y_test, dec.predict(x_test)), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_, annot=True,annot_kws={"size": 16}, fmt='g')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

"""**Random Forest**"""

r_for = RandomForestClassifier()
r_for.fit(x_train,y_train)

r_acc=accuracy_score(y_test,r_for.predict(x_test))

print("Train Set Accuracy:"+str(accuracy_score(y_train,r_for.predict(x_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test,r_for.predict(x_test))*100))

plt.figure(figsize=(6,4))
df_ = pd.DataFrame(confusion_matrix(y_test, r_for.predict(x_test)), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_, annot=True,annot_kws={"size": 16}, fmt='g')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

"""**K-NN**



"""

k_nei = KNeighborsClassifier()
k_nei.fit(x_train,y_train)

k_acc = accuracy_score(y_test,k_nei.predict(x_test))

print("Train set Accuracy:"+str(accuracy_score(y_train,k_nei.predict(x_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test,k_nei.predict(x_test))*100))

plt.figure(figsize=(6,4))
df_ = pd.DataFrame(confusion_matrix(y_test, k_nei.predict(x_test)), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_, annot=True,annot_kws={"size": 16}, fmt='g')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

"""**SVC**"""

s_vec = SVC()
s_vec.fit(x_train,y_train)

s_acc = accuracy_score(y_test,s_vec.predict(x_test))

print("Train set Accuracy:"+str(accuracy_score(y_train,s_vec.predict(x_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test,s_vec.predict(x_test))*100))

plt.figure(figsize=(6,4))
df_ = pd.DataFrame(confusion_matrix(y_test, s_vec.predict(x_test)), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_, annot=True,annot_kws={"size": 16}, fmt='g')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

"""**GB**"""

g_clf = GaussianNB()
g_clf.fit(x_train,y_train)

g_acc = accuracy_score(y_test,g_clf.predict(x_test))

print("Train set Accuracy:"+str(accuracy_score(y_train,g_clf.predict(x_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test,g_clf.predict(x_test))*100))

plt.figure(figsize=(6,4))
df_ = pd.DataFrame(confusion_matrix(y_test, g_clf.predict(x_test)), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_, annot=True,annot_kws={"size": 16}, fmt='g')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

"""**ANN**"""

# from keras.datasets import mnist

from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Dense

from keras.utils import np_utils
# from keras.layers import Dropout,Flatten

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]


# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(30,input_dim =7,activation="relu"))
    model.add(Dense(45,activation="tanh"))
    model.add(Dense(30,activation="relu"))
    model.add(Dense(45,activation="tanh"))
    model.add(Dense(35,activation="relu"))
    model.add(Dense(num_of_classes,activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=15,epochs=40)


# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=1)
print("train_Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set 

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print("test_Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

models = pd.DataFrame({'Model': ['Logistic','KNN', 'SVC',  'Decision Tree Classifier',
                       'Random Forest Classifier',  'Gaussian'],
                       'Accuracy': [ log_acc,k_acc, s_acc, dec_acc, r_acc, g_acc]})

models.sort_values(by = 'Accuracy', ascending = False) 

plt.figure(figsize = (16,3))
sns.barplot(x = 'Model', y = 'Accuracy', data = models)
plt.show()

"""We notice here that Logistic Regression is giveng us best accuracy result
 so ,we will go with the Logistic Regression model"""

log_reg = LogisticRegression()
log_reg.fit(X, Y)

test_y = pd.read_csv('C:/Data Science/Data Science/Titanic/gender_submission.csv')
print(test_y.head())

Y_final = log_reg.predict(test_data)
test_y['Y'] = Y_final
test_y.head()

log_reg = accuracy_score(test_y.Survived, test_y.Y)
print("Test set Accuracy:", log_reg)

plt.figure(figsize = (6,4)) 
df_ = pd.DataFrame(confusion_matrix(test_y.Survived, test_y.Y), range(2), range(2)) 
sns.set(font_scale = 1.4) #for label size
sns.heatmap(df_, annot = True, annot_kws = {"size": 16}, fmt = 'g')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

''''We Notice that although the accuracy was good for the test and train data
the same is not viable for the final case'''

'''Let us try running the second best model which was the SVC'''

s_vec = SVC()
s_vec.fit(X, Y)

test_y= pd.read_csv('C:/Data Science/Data Science/Titanic/gender_submission.csv')
print(test_y.head())

Y_final= s_vec.predict(test_data)
test_y['Y'] = Y_final
test_y.head()

s_acc = accuracy_score(test_y.Survived, test_y.Y)
print("Test set Accuracy:", s_acc)

plt.figure(figsize=(6,4))
df_ = pd.DataFrame(confusion_matrix(test_y.Survived, test_y.Y), range(2), range(2))
sns.set(font_scale = 1.4)#for label size
sns.heatmap(df_, annot = True, annot_kws = {"size": 16}, fmt = 'g')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

