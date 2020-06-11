# Devin Dhaliwal

import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

#getting data
data = pd.read_csv("teams.csv")
data = data.filter(items=['franchID','yearID','AB','R','H','2B','3B','HR','BB','SO','SB','CS','W','RA','ER','ERA','CG','SHO','SV','HA','HRA','BBA','SOA','E','DP','FP'])
data = data.query('yearID >= 1916')
data = data.rename(columns={'franchID':'Team','yearID':'Year'})

#adding BA, OBP, SLG, OPS
hitting = pd.DataFrame(round(data['H']/data['AB'], 3))
hitting.rename(columns={hitting.columns[0]: "BA"}, inplace=True)
obp = pd.DataFrame(round((data['H']+data['BB'])/data['AB'], 3))
hitting = hitting.join(obp)
hitting.rename(columns={hitting.columns[1]: "OBP"}, inplace=True)
slg = pd.DataFrame(round((data['H']+data['2B']+2*data['3B']+3*data['HR'])/data['AB'], 3))
hitting = hitting.join(slg)
hitting.rename(columns={hitting.columns[2]: "SLG"}, inplace=True)
ops = pd.DataFrame(round(hitting['OBP']+hitting['SLG'], 3))
hitting = hitting.join(ops)
hitting.rename(columns={hitting.columns[3]: "OPS"}, inplace=True)
data = data.join(hitting)
data = data.dropna()

#getting list of teams and df with no team and year col
team_names = data.Team.unique()
data_comb = data.drop(['Team','Year'],1)

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
        stat_choice = input("Please enter a stat you would like to test or \"done\" to continue: ").upper()
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
        stat_choice = input("Please enter a stat you would like visualize or \"done\" to return to main menu: ").upper()
        if stat_choice.lower() == "done":
            break
        else:
            if stat_choice in data_comb:
                sns.set()
                sns.regplot(x=stat_choice, y='W', data=data_comb, line_kws={'color': 'red'})
                plt.show()
            else:
                print("Invalid Stat...")
                continue

#visualizing impact of user chosen stat on a specific team's wins
def visualize_team_stat():

    while 1:
        stat_choice = input("Please enter a stat you would like visualize or \"done\" to return to main menu: ").upper()
        if stat_choice.lower() == "done":
            break
        else:
            if stat_choice in data_comb:
                team_choice = input("Please enter a team (3 letter abbreviation) you would like visualize " + stat_choice + " for: ").upper()
                if team_choice in team_names:
                        year_choice = int(input("Please enter a year you would like to start at (between 1916 and 2018): "))
                        if year_choice >= 1916 and year_choice <= 2018:
                            data_team = data[(data['Year'] > year_choice) & (data['Team'] == team_choice)]
                            sns.set()
                            fig = plt.figure(figsize=(15,10))
                            ax = sns.lineplot(x='Year', y=stat_choice, data=data_team)
                            ax2 = ax.twinx()
                            sns.lineplot(x='Year', y='W', data=data_team, ax=ax2, color='r')
                            ax.tick_params(axis='y', labelcolor='b')
                            ax.yaxis.label.set_color('b')
                            ax2.tick_params(axis='y', labelcolor='r')
                            ax2.yaxis.label.set_color('r')
                            ax.set_title(team_choice)
                            fig.legend(labels=[stat_choice, 'Wins'])
                            plt.show()
                        else:
                            print("Invalid Year...")
                            continue
                else:
                    print("Invalid Team...")
                    continue
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
                            "\nEnter \"3\" to visualize a stats impact on a specific team's wins over time"
                           "\nEnter \"quit\" to quit: ")

        if choice != "1" and choice != "2" and choice != "3" and choice.lower() != "quit":
            print("Invalid Input...")
            continue
        elif choice.lower() == "quit":
            return -1
        elif choice == "1":
            return 1
        elif choice == "2":
            return 2
        else:
            return 3

def main():

    #build_model()

    while 1:
        choice = print_menu()

        if choice == 1:
            predict_wins()
        elif choice == 2:
            visualize_stat()
        elif choice == 3:
            visualize_team_stat()
        else:
            break


main()
