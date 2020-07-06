# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#clear environment
from IPython import get_ipython;   
get_ipython().magic('reset -sf')
from IPython import get_ipython; 

#import packages for data analysis 
import pandas as pd
import os
import numpy as np 
import matplotlib.pyplot as plt 

#change directory to directory with data
os.chdir('D:\\a_Desktops_Git\\Current\\SpringBoard\\ultimate_challenge')

#import file
logins = pd.read_json("logins.json")

#convert index to date time
logins.index = logins.iloc[:,0]

#aggregate by 15 minutes
logins_15m = logins.resample("15T").count()

#aggregation of 1 hour intervals
mx = logins_15m.resample("1H").max()
logins_1h =pd.DataFrame(mx)
logins_1h['tmax'] = mx
logins_1h = logins_1h.drop(columns='login_time')
logins_1h['tmin'] = logins_15m.resample("1H").min()
logins_1h['tmn'] = logins_15m.resample("1H").mean()
logins_1h['tmd'] = logins_15m.resample("1H").median()
logins_1h['hmd'] = logins_1h.index.astype(str).str.split().str[1]

#exploratory plotting
#plt.plot(logins_1h['tmax'])
#plt.plot(logins_1h['tmin'])
#plt.plot(logins_1h['tmn'])
#plt.plot(logins_1h['tmd'])
#plt.xticks(rotation=45)
#plt.show()

#getting median of medians
logins_1hpd = logins_1h[['hmd','tmd']]
logins_1hpd = logins_1hpd.groupby('hmd').agg(['median'])

#exmaining data for plotting purposes
#logins_1h.tmd.iloc[4:28] #last = 2427

#code to plot
for i in list(range(0,100)):
        plt.plot(logins_1h.hmd.iloc[(4+24*i):(28+24*i)], 
                 logins_1h.tmd.iloc[(4+24*i):(28+24*i)], alpha = 0.08, 
                 c = 'blue')
        
plt.plot(logins_1hpd.index, logins_1hpd.tmd, c = 'red', linewidth=1.4)
plt.xticks(logins_1hpd.index[::2], rotation=45)
plt.xlabel('Time of day (hours)')
plt.ylabel('Median logins per 15 minutes per hour')
plt.show()

#Create a jupiter notebook
#part 2 

# 1) I might examine a few measures. The first measure I would be interested
# in looking at might be the typical range of drivers before and after 
# the implementation of the policy. Another measure I might examine would be
# the number of times a particular driver crossed the toll bridge in one day

# 2) a) I would select several drivers at random from Ultimate Gotham and 
# Ultimate Metropolis and partition them into control and experimental groups. 
# The control would keep driving as usual and the 
# experimental group would have all of their tolls paid for.  I might run the
# experiment for 1 to 1.5 months to get enough data and let the drivers become
# accustomed to the new change. 

# b) I am interested to determine if a change occures between two groups. 
# These kinds of tests lend themselves well to T-tests. 

# c) The interpretation of the results and the caveats of course depend on the
# specifics of the acquired data. If for instance their is a clear signal of 
# increased toll usage among drivers who have their tolls reimbursed, for 
# example a resulting p-value of 0.000001 or lower on a resulting T-test, this is a very
# clear signal that the change incentived drivers to use the toll road. There 
# may also be outliers that may need to be accounted for and interpreted as well.

#import file
UD = pd.read_json("s.json") #UD for Ultimate Data

# Ultimate is interested in predicting rider retention
# Ultimate defines retention as retained if the user took a trip in the last 
# 30 days

# how many unique values are present in each column?

# introductory exploration for cleaning purposes
list(set(list(UD.city))) #returns 3 cities
len(list(set(list(UD.trips_in_first_30_days)))) #59
len(list(set(list(UD.signup_date)))) # 31
len(list(set(list(UD.avg_rating_of_driver)))) # 8159 #several nan
len(list(set(list(UD.avg_surge)))) #115
len(list(set(list(UD.last_trip_date)))) # 82
len(list(set(list(UD.surge_pct)))) # 367
len(list(set(list(UD.ultimate_black_user)))) # 2
len(list(set(list(UD.weekday_pct)))) #666
len(list(set(list(UD.avg_dist)))) # 2908
len(list(set(list(UD.avg_rating_by_driver)))) # 228 #several nan
len(list(set(list(UD.phone)))) # 3 # None may be problematic

# create a retained column 
UD.last_trip_date.max() #'2014-07-01'
UD.last_trip_date.min() #'2014-01-01'

#move signup_date and last_trip_date to datetime
UD['signup_date'] = pd.to_datetime(UD.signup_date)
UD['last_trip_date'] = pd.to_datetime(UD.last_trip_date)

#create date for filtering
m30 = UD.last_trip_date.max() - pd.Timedelta('30 days') #m30 for minus 30 days

#create retained colummn
ret = UD.last_trip_date >= m30

#preliminary examination of retained vs lost users
UDrt = UD[ret == True]
UDrf = UD[ret == False]

#examination of some averages
UDrt.weekday_pct.mean() # pretty similar
UDrf.weekday_pct.mean()

UDrt.avg_dist.mean()
UDrf.avg_dist.mean() # pretty similar
 
UDrt.surge_pct.mean() # pretty similar
UDrf.surge_pct.mean()

UDrt.avg_surge.mean()
UDrf.avg_surge.mean()

UDrt.avg_rating_of_driver.mean()
UDrf.avg_rating_of_driver.mean()

UDrt.avg_rating_by_driver.mean()
UDrf.avg_rating_by_driver.mean()

UDrt.trips_in_first_30_days.mean() #
UDrf.trips_in_first_30_days.mean()

UDrt.phone[UDrt.phone == 'iPhone'].count()/len(UDrt.phone) #
UDrf.phone[UDrf.phone == 'iPhone'].count()/len(UDrf.phone)

UDrt.city[UDrt.city == "King's Landing"].count()/len(UDrt.city) #
UDrf.city[UDrf.city == "King's Landing"].count()/len(UDrf.city)

UDrt.city[UDrt.city == "Astapor"].count()/len(UDrt.city) 
UDrf.city[UDrf.city == "Astapor"].count()/len(UDrf.city) #

UDrt.city[UDrt.city == "Winterfell"].count()/len(UDrt.city) 
UDrf.city[UDrf.city == "Winterfell"].count()/len(UDrf.city) #
 
UDrt.ultimate_black_user[UDrt.ultimate_black_user == True].count()/len(UDrt.city) #
UDrf.ultimate_black_user[UDrf.ultimate_black_user == True].count()/len(UDrf.city)

# filter none from UD - none represents less 1 % of the data
UD = UD[(UD.phone == 'iPhone') | (UD.phone == 'Android')]

#create retained colummn
UD['ret'] = UD.last_trip_date >= m30

# replacing nan in avg_rating_of_driver and avg_rating_by_driver
UD.avg_rating_by_driver = UD.avg_rating_by_driver.fillna(UD.avg_rating_by_driver.median())
UD.avg_rating_of_driver = UD.avg_rating_of_driver.fillna(UD.avg_rating_of_driver.median())

# 1) What proportion of users were retained?
prop_ret = len(UD.ret[UD.ret == True])/len(UD.ret)

#    about 37.6 % of users were retained

# 2) Build a predictive model to help determine whether or not a user will be
#    active in 6 months. 

# creating numeric columns for ultimate_black_user, city. phone
cleanup = {"ultimate_black_user": {True: 1, False: 0},
           "city": {"King's Landing": 1, "Astapor": 2, "Winterfell": 3},
           "phone": {"iPhone":1, "Android":2}}

UD.replace(cleanup, inplace=True)

# doing a preliminary PCA analysis 
from sklearn.decomposition import PCA
from sklearn import preprocessing

# I think important factors for clustering are phone, city, 
# trips_first 30 days, surge, #avg_surge, avg_dist, weekday_pct

cUD = UD[['trips_in_first_30_days', 'surge_pct', 'avg_surge', 'avg_dist',
          'weekday_pct', 'phone', 'city']]

# normalize data
ncUD = pd.DataFrame(preprocessing.scale(cUD),columns = cUD.columns) 

#transpose data
tncUD = pd.DataFrame.transpose(ncUD) 

#fit the data
pca = PCA()

pca.fit_transform(tncUD)

#what are the important features of the PCA?
components = pd.DataFrame(pca.components_.transpose(), columns = ncUD.columns)

T = pca.transform(tncUD)

#creating color vector
ret_num = {"ret":{True: 1, False: 0}}
UD.replace(ret_num, inplace=True)

#plot prelabeling and after labeling
plt.scatter(pca.components_[0,:],pca.components_[1,:], alpha = 0.3, c = UD.ret)
plt.xlabel('PCA 1 ({} % of variance)'.format((pca.explained_variance_ratio_[0]*100).round()))
plt.ylabel('PCA 2 ({} % of variance)'.format((pca.explained_variance_ratio_[1]*100).round()))
plt.text(0, -0.070, 'Yellow: Active, Purple: Inactive')
plt.title('PCA of Ultimate Data')
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Create training and test sets
ncUD_train, ncUD_test, ret_train, ret_test = train_test_split(ncUD, UD.ret, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(ncUD_train, ret_train)

# Predict the labels of the test set: y_pred
ret_pred = logreg.predict(ncUD_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(ret_test, ret_pred))
print(classification_report(ret_test, ret_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

ncUD_train, ret_train = make_classification(n_samples=1000, n_features=7,
                           n_informative=2, n_redundant=0,
                           random_state=42, shuffle=False)
clf = RandomForestClassifier(max_depth=3, random_state=42)
clf.fit(ncUD_train, ret_train)
RandomForestClassifier(...)
rfpred = clf.predict(ncUD_test)

print(confusion_matrix(ret_test, rfpred))
print(classification_report(ret_test, rfpred))

# 2) I used both logistic regression and random forests to predict which users
#    would be active in 6 months. These models each do well with classification tasks. 
#    Both performed with similar accuracy,
#    although the random forest did a little better at predicting which users
#    were likely to be retained. The logistic regression shows that trips 
#    taken in the first 30 days, a user's phone, and the city they are located
#    are major predictors in user retention. These facors also agree with the
#    with the factors that showed the greatest difference in user retention 
#    from simple pairwise comparisons. 

# 3) There are several ways Ultimate could leverage this data. However since
#    users from King's landing, and with iPhone's tend to use the service more
#    Ultimate could leverage this information to attract users with this profile.