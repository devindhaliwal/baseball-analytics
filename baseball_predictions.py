# Devin Dhaliwal

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
pd.options.mode.chained_assignment = None

class dataset():
    
    #building dataset
    def __init__(self):
        #getting hitting avgs data
        data = pd.read_csv("baseball.csv")
        data = data.filter(items=['Team','Year','W','OBP','SLG','BA'])
        
        #adding OPS
        OPS_data = data[['OBP','SLG']]
        sum_col = OPS_data['OBP'] + OPS_data['SLG']
        OPS_data['OPS'] = sum_col
        OPS_data = OPS_data.drop(['OBP','SLG'],1)
        data = pd.merge(data, OPS_data, left_index=True,right_index=True)
        
        #getting other hitting and pitching data
        data_1 = pd.read_csv("teams.csv")
        data_1 = data_1.filter(items=['teamID','yearID','R','H','2B','3B','HR','BB','SO','SB','CS','W','RA','ER','ERA','CG','SHO','SV','HA','HRA','BBA','SOA','E','DP','FP'])
        data_1 = data_1.query('yearID >= 1962')
        data_1 = data_1.rename(columns={'teamID':'Team','yearID':'Year'})
        
        #combining hitting and pitching data
        self.data_comb = pd.merge(data,data_1,on=['Team','Year'])
        self.data_comb = self.data_comb.rename(columns={'W_x':'W'})
        
        #dropping uneeded data after combining
        self.data_comb = self.data_comb.drop(['Team','Year','W_y'],1)
        #saving non-normalized data for graph display
        self.data_norm = self.data_comb
        #saving wins data
        self.data_wins = self.data_comb[['W']]
    
    #normalizing features
    def normalize(self):
        scalar = MinMaxScaler()
        self.data_comb = scalar.fit_transform(self.data_comb.drop(['W'],1))        
        #putting data back into dataframe
        self.data_comb = pd.DataFrame(data=self.data_comb, columns=['OBP','SLG','BA','OPS','R','H','2B','3B','HR','BB','SO','SB','CS','RA','ER','ERA','CG','SHO','SV','HA','HRA','BBA','SOA','E','DP','FP'])
        self.data_comb = pd.merge(self.data_wins, self.data_comb, left_index=True,right_index=True)

    #splitting data into training and test
    def split_data(self):
        #getting attribute and label arrays for combined data
        attr_comb = np.array(self.data_comb.drop(['W'],1))
        labl_comb = np.array(self.data_wins)
        #splitting data into training and test
        self.attr_train, self.attr_test, self.labl_train, self.labl_test = sklearn.model_selection.train_test_split(attr_comb,labl_comb, test_size=0.2)
    
    #training model to be saved
    def train_model(self):
        best_acc = 0
        #saving best model out of 100 attempts
        for i in range(1000):
            
            #using linear regression
            linear = linear_model.LinearRegression()
            
            #training model
            linear.fit(self.attr_train, self.labl_train)
            
            #accuracy of model
            acc_comb = linear.score(self.attr_test, self.labl_test)
            print("COMBINED STATS:",acc_comb*100, "% accuracy")
            
            #saving model
            if acc_comb > best_acc:
                with open("baseballwinpredictionmodel.pickle", "wb") as f:
                    pickle.dump(linear, f)
                    
#loading saved model
def load_model():
    pickle_in = open("baseballwinpredictionmodel.pickle", "rb")
    model = pickle.load(pickle_in)
    return model

#getting user stats
#def user_stats():
    
    
#generating predictions
def predict_wins(model, data):
    predictions = model.predict(data)
    return predictions

#showing predicted wins vs actual wins for known data
def compare_predictions(predictions, labl_test):
    for i in range(len(predictions)):
        print("Predicted Wins:", round(predictions[i]), "\tActual Wins:", labl_test[i])

#showing prediction for user inputted stat
def display_prediction(predictions):
    for i in range(len(predictions)):
        print("Predicted Wins:", round(predictions[i]))

#accuracy of model
def show_model_accuracy(model, attr_comb_test, labl_comb_test):
    acc = model.score(attr_comb_test, labl_comb_test)
    print("\nACCURACY:",acc*100, "PERCENT")

#printing menu
def print_menu(data):
    print("STAT OPTIONS:")
    for stat in data:
        if stat != 'W':
            print(stat)
    print("QUIT")

#visualizing specific results
def visualize_stat(data):
    #checking which stat to visualize
    while 1:
        x_attr = input("What stat would you like to see: ").lower()
        if x_attr == "QUIT":
            return True
        if x_attr in data:
            break
        else:
            print("Invalid Stat, Please Try Again.")

    #displaying graph with attribute relating to wins
    style.use("ggplot")
    plt.scatter(data[x_attr],data['W'])
    plt.xlabel(x_attr)
    plt.ylabel('Wins')
    z = np.polyfit(data[x_attr],data['W'],1)
    p = np.poly1d(z)
    plt.plot(data[x_attr],p(data[x_attr]), "g--")
    plt.show()

def main():
    
    data = dataset()
    data.normalize()
    data.split_data()
    #data.train_model()
    model = load_model()
    predictions = predict_wins(model, data.attr_test)
    compare_predictions(predictions, data.labl_test)
    #display_prediction(predictions)
    show_model_accuracy(model, data.attr_test, data.labl_test)
    print_menu(data.data_norm)
    while 1:
        quit = visualize_stat(data.data_norm)
        if quit:
            break

main()
    
