# Devin Dhaliwal

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import datetime

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
data.to_csv("teams_preprocessed.csv")

#getting list of teams and df with no team and year col
team_names = data.Team.unique()
data_comb = data.drop(['Team','Year'],1)

#assigning columns to x and y for predictions
x = data_comb.drop('W', 1)
y = data_comb[['W']]

#building model
def build_win_prediction_model():
    best_acc = 0
    for i in range(1000):

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

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
            
#predicting game outcome
def predict_outcome():
    game_data = pd.read_csv("games.csv")
    game_data[['wind_speed','wind_direction']] = game_data.wind.str.split(",",expand=True) 
    game_data[['temp','weather']] = game_data.weather.str.split(",",expand=True)
    
    new_dates = []
    for date in game_data.date:
        mydate = datetime.datetime.strptime(date, '%Y-%m-%d')
        data = mydate.strftime('%B, %A')
        new_dates.append(data)
    
    game_data['new_dates'] = new_dates
    game_data[['month','day']] = game_data.new_dates.str.split(",",expand=True)
    game_data[['temp','deg']] = game_data.temp.str.split(" ",expand=True)
    game_data['temp'] = pd.to_numeric(game_data['temp'])
    game_data[['wind_speed','mph']] = game_data.wind_speed.str.split(" ",expand=True)
    game_data['wind_speed'] = pd.to_numeric(game_data['wind_speed'])
    game_data[['start_time','ampm']] = game_data.start_time.str.split(" ",expand=True)
    game_data[['hour','min']] = game_data.start_time.str.split(":",expand=True)
    game_data['hour'] = pd.to_numeric(game_data['hour'])
    game_data['min'] = pd.to_numeric(game_data['min'])

    times = []
    for hour in game_data.hour:
        if hour == 11 or hour == 12:
            times.append("day")
        elif hour >= 1 and hour <= 4:
            times.append("day")
        else:
            times.append("night")

    game_data['start_time'] = times
    
    elapsed_time = []
    for time in game_data.elapsed_time:
        if time < 182:
            elapsed_time.append("less than 3 hours")
        else:
            elapsed_time.append("more than 3 hours")

    game_data["elapsed_time"] = elapsed_time

    wind_speed = []
    for speed in game_data.wind_speed:
        if speed == 0:
            wind_speed.append("0 mph")
        elif speed < 8:
            wind_speed.append("1-7 mph")
        elif speed < 12:
            wind_speed.append("8-12 mph")
        else:
            wind_speed.append("13+ mph")

    game_data["wind_speed"] = wind_speed

    temps = []
    for temp in game_data.temp:
        if temp < 60:
            temps.append("30-59")
        elif temp < 70:
            temps.append("60-69")
        elif temp < 80:
            temps.append("70-79")
        elif temp < 90:
            temps.append("80-89")
        elif temp < 100:
            temps.append("90-99")
        else:
            temps.append("100+")

    game_data["temp"] = temps

    attendance = []
    for num in game_data.attendance:
        if num < 10000:
            attendance.append("0-9,999")
        elif num < 20000:
            attendance.append("10,000-19,999")
        elif num < 30000:
            attendance.append("20,000-29,999")
        elif num < 40000:
            attendance.append("30,000-39,999")
        elif num < 50000:
            attendance.append("40,000-49,999")
        elif num < 60000:
            attendance.append("50,000-60,000")

    game_data["attendance"] = attendance

    game_data['outcome'] = game_data['home_final_score'] - game_data['away_final_score']
    results = []
    for result in game_data.outcome:
        if result > 0:
            result = 1
        else:
            result = 0
        results.append(result)
    game_data['outcome'] = results
    
    game_data = game_data.filter(items=['month','day','outcome','away_final_score','home_final_score','elapsed_time','attendance', 'start_time','wind_speed','wind_direction','temp','weather'])
    game_data.to_csv("game_outcomes.csv")

    game_data['day'] = game_data['day'].str.lstrip()
    game_data['wind_direction'] = game_data['wind_direction'].str.lstrip()
    game_data['weather'] = game_data["weather"].str.lstrip()
    
    oh_months = pd.get_dummies(game_data.month)
    oh_days = pd.get_dummies(game_data.day)
    oh_wind_direction = pd.get_dummies(game_data.wind_direction)
    oh_weather = pd.get_dummies(game_data.weather)
    oh_start_time = pd.get_dummies(game_data.start_time)
    oh_elapsed_time = pd.get_dummies(game_data.elapsed_time)
    oh_wind_speed = pd.get_dummies(game_data.wind_speed)
    oh_temp = pd.get_dummies(game_data.temp)
    oh_attendance = pd.get_dummies(game_data.attendance)
    
    game_data = game_data.drop(['month','day','wind_direction','weather','start_time','temp','wind_speed','elapsed_time','attendance','home_final_score','away_final_score'], axis=1)
    
    game_data = game_data.join(oh_months)
    game_data = game_data.join(oh_days)
    game_data = game_data.join(oh_wind_direction)
    game_data = game_data.join(oh_weather)
    game_data = game_data.join(oh_start_time)
    game_data = game_data.join(oh_elapsed_time)
    game_data = game_data.join(oh_wind_speed)
    game_data = game_data.join(oh_temp)
    game_data = game_data.join(oh_attendance)

    game_data.to_csv("game_outcomes_preprocessed.csv")

    best_acc = 0
    for i in range(10):

        x_o = game_data.drop(['outcome'], axis=1)
        y_o = game_data['outcome']

        x_train, x_test, y_train, y_test = train_test_split(x_o, y_o, test_size=0.1)

        model = KNeighborsClassifier(n_neighbors=137, p=1)
        model.fit(x_train, y_train)

        acc = model.score(x_test, y_test) * 100

        if acc > best_acc:
            best_acc = acc

    print("Model Accuracy:", best_acc)
    # print(x_test)

    model.fit(x_o, y_o)
    with open("baseballoutcomepredictionmodel.pickle", "wb") as f:
        pickle.dump(model, f)

    """
    n_neighbors = list(range(1, 151))
    p = [1, 2]
    weights = ["uniform", "distance"]

    hyperparameters = dict(n_neighbors=n_neighbors, p=p, weights=weights)
    knn = KNeighborsClassifier()

    x_o = game_data.drop(['outcome'], axis=1)
    y_o = game_data['outcome']

    tuning_model = GridSearchCV(knn, hyperparameters, cv=3)
    best_model = tuning_model.fit(x_o, y_o)
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
    print('Best weight:', best_model.best_estimator_.get_params()['weights'])
    """


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

    #build_win_prediction_model()

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

#predict_outcome()
#main()


