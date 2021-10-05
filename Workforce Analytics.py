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
stacked_plot(df_company, "Higher_Education", "Attrition")
stacked_plot(df_company, "Status_of_leaving", "Attrition")
stacked_plot(df_company, "Mode_of_work", "Attrition")
stacked_plot(df_company, "leaves", "Attrition")
stacked_plot(df_company, "absenteeism", "Attrition")
stacked_plot(df_company, "Work_accident", "Attrition")
stacked_plot(df_company, "Source_of_Hire", "Attrition")
stacked_plot(df_company, "Job_mode", "Attrition")


# we plot the heat map to see the various relationships

plt.figure(figsize = (10,8))
sns.heatmap(df_company.corr(), annot = False, cmap = 'coolwarm')
plt.show()

corr_attr = df_company.corr()
(corr_attr['Attrition'].sort_values(ascending = False))

col = df_company.corr().nlargest(20, "Attrition").Attrition.index
plt.figure(figsize=(15, 15))
sns.heatmap(df_company[col].corr(), annot = True, cmap = "RdYlGn", annot_kws = {"size":10})

# Let us try and calculate chi-values 
from sklearn.feature_selection import chi2
X = df_company.drop('Attrition', axis = 1)
y = df_company['Attrition']
chi_scores = chi2(X, y)
chi_scores

# Here first array represents chi square values and second array represents p-values
# and plotting the values as per their values will show the importance or impact on the attrition
p_values = pd.Series(chi_scores[1], index = X.columns)
p_values.sort_values(ascending = False, inplace = True)
p_values.plot.bar()




