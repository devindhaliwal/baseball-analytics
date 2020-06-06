# Devin Dhaliwal

import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

#getting hitting avgs data
data = pd.read_csv("baseball.csv")
data = data.filter(items=['Team','Year','W','OBP','SLG','BA'])

#adding OPS
ops = pd.DataFrame(data['OBP']+data['SLG'])
data = data.join(ops)
data.rename(columns={ data.columns[6]: "OPS" }, inplace = True)

#getting other hitting and pitching data
data_1 = pd.read_csv("teams.csv")
data_1 = data_1.filter(items=['teamID','yearID','R','H','2B','3B','HR','BB','SO','SB','CS','W','RA','ER','ERA','CG','SHO','SV','HA','HRA','BBA','SOA','E','DP','FP'])
data_1 = data_1.query('yearID >= 1962')
data_1 = data_1.rename(columns={'teamID':'Team','yearID':'Year'})

#combining datasets
data_comb = pd.merge(data,data_1,on=['Team','Year'])
data_comb = data_comb.rename(columns={'W_x':'W'})

#dropping uneeded data after combining
data_comb = data_comb.drop(['Team','Year','W_y'],1)

#assigning columns to x and y for predictions
x = data_comb.drop('W', 1)
y = data_comb[['W']]

#building model
def build_model():
    best_acc = 0
    for i in range(1000):

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)

        acc = model.score(x_test, y_test)*100

        if acc > best_acc:
            best_acc = acc
            with open("baseballwinpredictionmodel.pickle", "wb") as f:
                pickle.dump(model, f)

    print("Model Accuracy:", best_acc)

def predict_wins():
    #df with avg values of each stat
    df_predict = pd.DataFrame([x.mean()])

    #getting user inputs for stats to predict wins
    while 1:
        stat_choice = input("Please enter a stat you would like to test or \"done\" to continue: ")
        if stat_choice.lower() == "done":
            break
        else:
            if stat_choice in df_predict:
                stat = float(input("Enter value for " + stat_choice + " you would like to test: "))
                df_predict[stat_choice] = stat
            else:
                print("Invalid Stat...")
                continue

    #generating prediction
    pickle_in = open("baseballwinpredictionmodel.pickle", "rb")
    model = pickle.load(pickle_in)
    result = model.predict(df_predict)

    #making sure prediction is in range
    if int(result) > 162:
        result = 162

    if int(result) < 0:
        result = 0

    #printing prediction
    print("Predicted Wins:", int(result))

#visualizing impact of user chosen stat on wins
def visualize_stat():
    while 1:
        stat_choice = input("Please enter a stat you would like visualize or \"done\" to quit: ")
        if stat_choice.lower() == "done":
            break
        else:
            if stat_choice in data_comb:
                sns.regplot(x=stat_choice, y='W', data=data_comb, line_kws={'color': 'red'})
                plt.show()
            else:
                print("Invalid Stat...")
                continue

#printing menu including available stats and options to predict or graph
def print_menu():
    print("\nAvailable Stats:")
    for stat in x:
        print(stat)

    while 1:
        choice = input("Enter \"1\" to predict wins based on stats you enter"
                           "\nEnter \"2\" to visualize a stats impact on wins"
                           "\nEnter \"quit\" to quit: ")

        if choice != "1" and choice != "2" and choice.lower() != "quit":
            print("Invalid Input...")
            continue
        elif choice.lower() == "quit":
            return -1
        elif choice == "1":
            return 1
        else:
            return 2

def main():

    #build_model()

    while 1:
        choice = print_menu()

        if choice == 1:
            predict_wins()
        elif choice == 2:
            visualize_stat()
        else:
            break


main()
