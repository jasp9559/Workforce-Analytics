'''Project work for Workforce Analytics, wherein we need to work on the available features 
   to figure out a solution on the attrition and forecast the problem in order to help 
   stabilise the attrition rate and sturdy the features to help the businiess flow be proper'''

'''We start by importing the data and having a look at the various features gathered for the problem'''

# Importing the libraires
import pandas as pd
import numpy as np


attrition = pd.read_excel("C:/Data Science/My work/HR Workforce Analytics Project/Final dataset Attrition.xlsx")
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

# Lets check out some visualisation to get the insights on the data
df_company = attrition

import seaborn as sns 
import matplotlib.pyplot as plt
def stacked_plot(df, group, target):
    """
    Function to generate a stacked plots between two variables
    """
    fig, ax = plt.subplots(figsize = (6,4))
    temp_df = (df.groupby([group, target]).size()/df.groupby(group)[target].count()).reset_index().pivot(columns=target, index=group, values=0)
    temp_df.plot(kind = 'bar', stacked = True, ax = ax, color = ["green", "darkred"])
    ax.xaxis.set_tick_params(rotation = 0)
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

''' 
Having a look at the plot which gives the relation between attrition and age group gives the insight that
the employees in the age group of under 25 tend to move faster and the ones within 25 and 32 also.
'''

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
    
df_company["DistanceFromHome_group"] = df_company["DistanceFromHome"].apply(lambda x: DistanceFromHome(x))
df_company["DistanceFromHome_group"].value_counts()
sns.countplot(x = "DistanceFromHome_group", hue="Attrition", data = df_company)

''' 
Now taking the relation between attrition and Distance from home gives the insight that 
the employees with a farther distance from home tend to take a decision to attrite quite obviously.
'''

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

''' 
Now this interesting fact is very well known that the one year atrrition employees are 
known as Jumpers but this does go against their profile, and then the most attritions 
take place in the range of 1 to 5 years of employment.
'''


'''
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

''' 
Now we try and visualise the factors that effect the attrtion most using the stacked plots as under.
Not only does it give a better understanding but the visuals help select the features better.
'''

df_company = attrition_mms 

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
stacked_plot(df_company, "Higher_Education", "Attrition")
stacked_plot(df_company, "Status_of_leaving", "Attrition")
stacked_plot(df_company, "Mode_of_work", "Attrition")
stacked_plot(df_company, "Leaves", "Attrition")
stacked_plot(df_company, "Absenteeism", "Attrition")
stacked_plot(df_company, "Work_accident", "Attrition")
stacked_plot(df_company, "Source_of_Hire", "Attrition")
stacked_plot(df_company, "Job_mode", "Attrition")

##############################################
# We plot the heat map to see the various relationships under correlation using the heatmap

plt.figure(figsize = (10,8))
sns.heatmap(df_company.corr(), annot = False, cmap = 'coolwarm')
plt.show()

# Checking the correlation coeficients and importance ordered
corr_attr = df_company.corr()
(corr_attr['Attrition'].sort_values(ascending = False))

col = df_company.corr().nlargest(20, "Attrition").Attrition.index
plt.figure(figsize=(15, 15))
sns.heatmap(df_company[col].corr(), annot = True, cmap = "RdYlGn", annot_kws = {"size":10})

##############################################
# Let us additionally reinforce the feature selection by trying to calculate chi-values
from sklearn.feature_selection import chi2
X = df_company.drop('Attrition', axis = 1)
y = df_company['Attrition']
chi_scores = chi2(X, y)
chi_scores

# Here first array represents chi square values and second array represents p-values
# and plotting the values as per their values will show the importance or impact on the attrition
p_values = pd.Series(chi_scores[1], index = X.columns)
p_values.sort_values(ascending = True, inplace = True)
p_values.plot.bar()

'''
Now off the plot bar which gives the impactful features stacked together ordered by their importance as under
We select the ones which create a significant impact on Attrition
'''


##############################################
'''
We use now various features that are impactful on the attrition and 
try to check the survival analysis over them to determine the duration
'''

import lifelines

df = pd.read_excel("C:/Data Science/My work/HR Workforce Analytics Project/Final dataset Attrition.xlsx")

# Taking "YearsAtCompany" to be time spell
T = df.YearsAtCompany 

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter
# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()
# Fitting KaplanMeierFitter model on Time and Events for Attrition
kmf.fit(durations = T, event_observed = df_company.Attrition)
# Time-line estimations plot 
kmf.survival_function_.plot()
plt.title('Survival curve wrt the Attrition as event and YearsAtCompany as spell')
plt.show()

# Print survival probabilities at each year
kmf.survival_function_

# Plot the survival function with confidence intervals
kmf.plot_survival_function()
plt.show()

##############################################
# We try over Multiple groups with the event being "Attrition"
''' We first select the group to be OverTime'''
df_company.OverTime.value_counts()

OT_worked = df_company.OverTime == 1
OT_Not = df_company.OverTime == 0
# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[df_company.OverTime == 1], df_company.Attrition[df_company.OverTime == 1], label = 'OT_worked')
ax = kmf.survival_function_.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[df_company.OverTime == 0], df_company.Attrition[df_company.OverTime == 0], label = 'OT_Not')
kmf.survival_function_.plot(ax=ax)
plt.title('Survival plot for "Attrition" w.r.t "OverTime"')

##############################################
''' We now select the group to be BusinessTravel'''
df_company.BusinessTravel.value_counts()

Frequent = df_company.BusinessTravel == 1.00
Rare = df_company.BusinessTravel == 0.50
Non = df_company.BusinessTravel == 0.00
# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[df_company.BusinessTravel == 1], df_company.Attrition[df_company.BusinessTravel == 1], label = 'Frequent')
ax = kmf.survival_function_.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0.5"
kmf.fit(T[df_company.BusinessTravel == 0.5], df_company.Attrition[df_company.BusinessTravel == 0.5], label = 'Rare')
kmf.survival_function_.plot(ax=ax)

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[df_company.BusinessTravel == 0], df_company.Attrition[df_company.BusinessTravel == 0], label = 'Non')
kmf.survival_function_.plot(ax=ax)
plt.title('Survival plot for "Attrition" w.r.t "BusinessTravel"')

##############################################
''' We now select the group to be JobLevel'''
df_company.JobLevel.value_counts()


# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[df_company.JobLevel == 1], df_company.Attrition[df_company.JobLevel == 1], label = '5')
ax = kmf.survival_function_.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0.75"
kmf.fit(T[df_company.JobLevel == 0.75], df_company.Attrition[df_company.JobLevel == 0.75], label = '4')
kmf.survival_function_.plot(ax=ax)

# Applying KaplanMeierFitter model on Time and Events for the group "0.50"
kmf.fit(T[df_company.JobLevel == 0.50], df_company.Attrition[df_company.JobLevel == 0.50], label = '3')
kmf.survival_function_.plot(ax=ax)

# Applying KaplanMeierFitter model on Time and Events for the group "0.25"
kmf.fit(T[df_company.JobLevel == 0.25], df_company.Attrition[df_company.JobLevel == 0.25], label = '2')
kmf.survival_function_.plot(ax=ax)

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[df_company.JobLevel == 0], df_company.Attrition[df_company.JobLevel == 0], label = '1')
kmf.survival_function_.plot(ax=ax)
plt.title('Survival plot for "Attrition" w.r.t "JobLevel"')

##############################################
''' We now select the group to be Age_group'''
df_company.Age_group.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[df_company.Age_group == 1], df_company.Attrition[df_company.Age_group == 1], label = '1')
ax = kmf.survival_function_.plot() 

# Applying KaplanMeierFitter model on Time and Events for the group "0.75"
kmf.fit(T[df_company.Age_group == 0.75], df_company.Attrition[df_company.Age_group == 0.75], label = '0.75')
kmf.survival_function_.plot(ax=ax) 

# Applying KaplanMeierFitter model on Time and Events for the group "0.50"
kmf.fit(T[df_company.Age_group == 0.50], df_company.Attrition[df_company.Age_group == 0.50], label = '0.50')
kmf.survival_function_.plot(ax=ax) 

# Applying KaplanMeierFitter model on Time and Events for the group "0.25"
kmf.fit(T[df_company.Age_group == 0.25], df_company.Attrition[df_company.Age_group == 0.25], label = '0.25')
kmf.survival_function_.plot(ax=ax) 

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[df_company.Age_group == 0], df_company.Attrition[df_company.Age_group == 0], label = '0')
kmf.survival_function_.plot(ax=ax) 
plt.title('Survival plot for "Attrition" w.r.t "Age_group"') 


#######################################################
'''
We start building the models for classification
We start by splitting the data into Train and test
'''
#######################################################

from sklearn.model_selection import train_test_split
df = df_company.iloc[:, 1]
df1 = df_company.drop('Attrition', axis = 1)
X = df1
Y = df

# herein we split the data with test size kept as 15%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 40)
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
    #Initializing the model 
    model = Sequential()
    model.add(Dense(50, input_dim = 31, activation = "relu"))
    model.add(Dense(65, activation = "tanh"))
    model.add(Dense(50, activation = "relu"))
    model.add(Dense(45, activation = "tanh"))
    model.add(Dense(35, activation = "relu"))
    model.add(Dense(num_of_classes, activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x = x_train, y = y_train, batch_size = 50, epochs = 100)


# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=1)
print("train_Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set 

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print("test_Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
ann_acc = eval_score_test[1]
# accuracy on test data set

models = pd.DataFrame({'Model': ['Logistic','KNN', 'SVC',  'Decision Tree Classifier',
                       'Random Forest Classifier',  'Gaussian', 'ANN'],
                       'Accuracy': [ log_acc,k_acc, s_acc, dec_acc, r_acc, g_acc, ann_acc]})

models.sort_values(by = 'Accuracy', ascending = False) 

plt.figure(figsize = (16,3))
sns.barplot(x = 'Model', y = 'Accuracy', data = models)
plt.show()

"""We notice here that Logistic Regression is giving us the best accuracy result
 so ,we will go with the Logistic Regression model"""

