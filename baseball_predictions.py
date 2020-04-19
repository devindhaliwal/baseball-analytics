import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

#getting hitting data
data = pd.read_csv("baseball.csv")
data = data.filter(items=['Team','Year','W','OBP','SLG','BA'])

#getting pitching data
data_1 = pd.read_csv("teams.csv")
data_1 = data_1.filter(items=['teamID','yearID','W','ERA','CG','SHO','SV','HA','HRA','BBA','SOA','E','DP','FP'])
data_1 = data_1.query('yearID >= 1962')
data_1 = data_1.rename(columns={'teamID':'Team','yearID':'Year'})

#combining hitting and pitching data
data_comb = pd.merge(data,data_1,on=['Team','Year'])
data_comb = data_comb.rename(columns={'W_x':'W'})

#dropping uneeded data after combining
#data = data.drop(['Team','Year'],1)
#data_1 = data_1.drop(['Team','Year'],1)
data_comb = data_comb.drop(['Team','Year','W_y'],1)

#normalizing features
scalar = MinMaxScaler()
#data_scaled = scalar.fit_transform(data)
#data1_scaled = scalar.fit_transform(data_1)
data_comb_scaled = scalar.fit_transform(data_comb.drop(['W'],1))

#putting data back into dataframe
#data_scaled_df = pd.DataFrame(data=data_scaled, columns=['W','OBP','SLG','BA'])
#data1_scaled_df = pd.DataFrame(data=data1_scaled, columns=['W','ERA','CG','SHO','SV','HA','HRA','BBA','SOA','E','DP','FP'])
data_comb_scaled_df = pd.DataFrame(data=data_comb_scaled, columns=['OBP','SLG','BA','ERA','CG','SHO','SV','HA','HRA','BBA','SOA','E','DP','FP'])

#adding non normalized wins back to dataframe
data_wins = data_comb[['W']]
data_comb_df = pd.merge(data_wins, data_comb_scaled_df, left_index=True,right_index=True)


"""
#using linear regression
linear = linear_model.LinearRegression()

#getting attribute and label arrays for hitting
attr = np.array(data_scaled_df.drop(['W'],1))
labl = np.array(data_scaled_df['W'])

#training model
attr_train, attr_test, labl_train, labl_test = sklearn.model_selection.train_test_split(attr,labl, test_size=0.2)
linear.fit(attr_train, labl_train)

#accuracy of model using only hitting data
acc = linear.score(attr_test, labl_test)
print("HITTING STATS:",acc*100, "% accuracy")

#getting attribute and label arrays for pitching
attr1 = np.array(data1_scaled_df.drop(['W'],1))
labl1 = np.array(data1_scaled_df['W'])

#training model
attr1_train, attr1_test, labl1_train, labl1_test = sklearn.model_selection.train_test_split(attr1,labl1, test_size=0.2)
linear.fit(attr1_train, labl1_train)

#accuracy of model for only pitching data
acc1 = linear.score(attr1_test, labl1_test)
print("PITCHING STATS:",acc1*100, "% accuracy")
"""

#getting attribute and label arrays for combined data
attr_comb = np.array(data_comb_df.drop(['W'],1))
labl_comb = np.array(data_comb_df['W'])

#splitting data into training and test
attr_comb_train, attr_comb_test, labl_comb_train, labl_comb_test = sklearn.model_selection.train_test_split(attr_comb,labl_comb, test_size=0.2)

"""
best_acc = 0
#saving best model out of 100 attempts
for i in range(100):

    #splitting data into training and test
    attr_comb_train, attr_comb_test, labl_comb_train, labl_comb_test = sklearn.model_selection.train_test_split(attr_comb,labl_comb, test_size=0.2)
    
    #using linear regression
    linear = linear_model.LinearRegression()
    
    #training model
    linear.fit(attr_comb_train, labl_comb_train)
    
    #accuracy of model
    acc_comb = linear.score(attr_comb_test, labl_comb_test)
    print("COMBINED STATS:",acc_comb*100, "% accuracy")
    
    #saving model
    if acc_comb > best_acc:
        with open("baseballwinpredictionmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

#loading saved model
pickle_in = open("baseballwinpredictionmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#generating predictions
predictions = linear.predict(attr_comb_test)

#showing predictions
for i in range(len(predictions)):
    print("Predicted Wins:", round(predictions[i]), "\tActual Wins:", labl_comb_test[i])

#accuracy of model
acc_comb = linear.score(attr_comb_test, labl_comb_test)
print("\nACCURACY:",acc_comb*100, "PERCENT")

#printing menu
#stats = ['OBP', 'SLG', 'BA', 'ERA','CG','SHO','SV','HA','HRA','BBA','SOA','E','DP','FP']
print("STAT OPTIONS:")
for stat in data_comb:
    if stat != 'W':
        print(stat)

#visualizing specific results
while 1:
    x_attr = input("What stat would you like to see: ")
    if x_attr in data_comb:
        break
    else:
        print("Invalid Stat, Please Try Again.")

style.use("ggplot")
plt.scatter(data_comb[x_attr],data_comb_df['W'])
plt.xlabel(x_attr)
plt.ylabel('Wins')
z = np.polyfit(data_comb[x_attr],data_comb_df['W'],1)
p = np.poly1d(z)
plt.plot(data_comb[x_attr],p(data_comb[x_attr]), "g--")
plt.show()






