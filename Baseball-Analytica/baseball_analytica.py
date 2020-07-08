import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import time
import matplotlib
import datetime
import boto3
import ssl
import io
ssl._create_default_https_context = ssl._create_unverified_context

matplotlib.use('Agg')

#getting data
url = "https://raw.githubusercontent.com/devindhaliwal/baseball-analytics/master/Baseball-Analytica/Teams.csv"
data = pd.read_csv(url, sep=",")
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
#data.to_csv("teams_preprocessed.csv")

# getting list of teams and df with no team and year col
team_names = data.Team.unique()
data_comb = data.drop(['Team', 'Year'], 1)

# assigning columns to x and y for predictions
x = data_comb.drop('W', 1)
y = data_comb[['W']]


def graph_stat(stat):
    if stat in data_comb:
        fig = plt.figure(figsize=(12, 10))
        sns.set()
        sns.regplot(x=stat, y='W', data=data_comb, line_kws={'color': 'red'})
        new_graph_name_im = "stat_graph_" + str(time.time()) + ".png"
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png')
        img_data.seek(0)
        image = img_data.read()
        s3 = boto3.resource('s3')
        s3.Object("baseball-analytica-graphs", new_graph_name_im).put(ACL='public-read', Body=bytes(image))
        link = "https://baseball-analytica-graphs.s3-us-west-1.amazonaws.com/"+new_graph_name_im
        return link
    else:
        return "Invalid Input"


def graph_team_stat(stat, team, year):
    if stat in data_comb:
        if team in team_names:
            if year >= 1916 and year <= 2017:
                data_team = data[(data['Year'] >= year) & (data['Team'] == team)]
                fig = plt.figure(figsize=(12, 10))
                sns.set()
                ax = sns.lineplot(x='Year', y=stat, data=data_team)
                ax2 = ax.twinx()
                sns.lineplot(x='Year', y='W', data=data_team, ax=ax2, color='r')
                ax.tick_params(axis='y', labelcolor='b')
                ax.yaxis.label.set_color('b')
                ax2.tick_params(axis='y', labelcolor='r')
                ax2.yaxis.label.set_color('r')
                ax.set_title(team)
                fig.legend(labels=[stat, 'Wins'])
                new_graph_name_im = "team_stat_graph_" + str(time.time()) + ".png"
                img_data = io.BytesIO()
                fig.savefig(img_data, format='png')
                img_data.seek(0)
                image = img_data.read()
                s3 = boto3.resource('s3')
                s3.Object("baseball-analytica-graphs", new_graph_name_im).put(ACL='public-read', Body=bytes(image))
                link = "https://baseball-analytica-graphs.s3-us-west-1.amazonaws.com/" + new_graph_name_im
                return link
            else:
                return "Invalid Input"
        else:
            return "Invalid Input"
    else:
        return "Invalid Input"


def predict_wins(AB, R, H, _2B, _3B, HR, BB, SO, SB, CS, RA, ER, ERA, CG, SHO,
                 SV, HA, HRA, BBA, SOA, E, DP, FP, BA, OBP, SLG, OPS):
    df_predict = pd.DataFrame([x.mean()])

    if AB == "":
        AB = x["AB"].mean()
    else:
        try:
            AB = int(AB)
            df_predict["AB"] = AB
        except ValueError:
            return "Invalid Input for AB"
    if R == "":
        R = x["R"].mean()
    else:
        try:
            R = int(R)
            df_predict["R"] = R
        except ValueError:
            return "Invalid Input for R"
    if H == "":
        H = x["H"].mean()
    else:
        try:
            H = int(H)
            df_predict["H"] = H
        except ValueError:
            return "Invalid Input for H"
    if _2B == "":
        _2B = x["2B"].mean()
    else:
        try:
            _2B = int(_2B)
            df_predict["2B"] = _2B
        except ValueError:
            return "Invalid Input for 2B"
    if _3B == "":
        _3B = x["3B"].mean()
    else:
        try:
            _3B = int(_3B)
            df_predict["3B"] = _3B
        except ValueError:
            return "Invalid Input for 3B"
    if HR == "":
        HR = x["HR"].mean()
    else:
        try:
            HR = int(HR)
            df_predict["HR"] = HR
        except ValueError:
            return "Invalid Input for HR"
    if BB == "":
        BB = x["BB"].mean()
    else:
        try:
            BB = int(BB)
            df_predict["BB"] = BB
        except ValueError:
            return "Invalid Input for BB"
    if SO == "":
        SO = x["SO"].mean()
    else:
        try:
            SO = int(SO)
            df_predict["SO"] = SO
        except ValueError:
            return "Invalid Input for SO"
    if SB == "":
        SB = x["SB"].mean()
    else:
        try:
            SB = int(SB)
            df_predict["SB"] = SB
        except ValueError:
            return "Invalid Input for SB"
    if CS == "":
        CS = x["CS"].mean()
    else:
        try:
            CS = int(CS)
            df_predict["CS"] = CS
        except ValueError:
            return "Invalid Input for CS"
    if RA == "":
        RA = x["RA"].mean()
    else:
        try:
            RA = int(RA)
            df_predict["RA"] = RA
        except ValueError:
            return "Invalid Input for RA"
    if ER == "":
        ER = x["ER"].mean()
    else:
        try:
            ER = int(ER)
            df_predict["ER"] = ER
        except ValueError:
            return "Invalid Input for ER"
    if ERA == "":
        ERA = x["ERA"].mean()
    else:
        try:
            ERA = float(ERA)
            df_predict["ERA"] = ERA
        except ValueError:
            return "Invalid Input for ERA"
    if CG == "":
        CG = x["CG"].mean()
    else:
        try:
            CG = int(CG)
            df_predict["CG"] = CG
        except ValueError:
            return "Invalid Input for CG"
    if SHO == "":
        SHO = x["SHO"].mean()
    else:
        try:
            SHO = int(SHO)
            df_predict["SHO"] = SHO
        except ValueError:
            return "Invalid Input for SHO"
    if SV == "":
        SV = x["SV"].mean()
    else:
        try:
            SV = int(SV)
            df_predict["SV"] = SV
        except ValueError:
            return "Invalid Input for SV"
    if HA == "":
        HA = x["HA"].mean()
    else:
        try:
            HA = int(HA)
            df_predict["HA"] = HA
        except ValueError:
            return "Invalid Input for HA"
    if HRA == "":
        HRA = x["HRA"].mean()
    else:
        try:
            HRA = int(HRA)
            df_predict["HRA"] = HRA
        except ValueError:
            return "Invalid Input for HRA"
    if BBA == "":
        BBA = x["BBA"].mean()
    else:
        try:
            BBA = int(BBA)
            df_predict["BBA"] = BBA
        except ValueError:
            return "Invalid Input for BBA"
    if SOA == "":
        SOA = x["SOA"].mean()
    else:
        try:
            SOA = int(SOA)
            df_predict["SOA"] = SOA
        except ValueError:
            return "Invalid Input for SOA"
    if E == "":
        E = x["E"].mean()
    else:
        try:
            E = int(E)
            df_predict["E"] = E
        except ValueError:
            return "Invalid Input for E"
    if DP == "":
        DP = x["DP"].mean()
    else:
        try:
            DP = int(DP)
            df_predict["DP"] = DP
        except ValueError:
            return "Invalid Input for DP"
    if FP == "":
        FP = x["FP"].mean()
    else:
        try:
            FP = float(FP)
            df_predict["FP"] = FP
        except ValueError:
            return "Invalid Input for FP"
    if BA == "":
        BA = x["BA"].mean()
    else:
        try:
            BA = float(BA)
            df_predict["BA"] = BA
        except ValueError:
            return "Invalid Input for BA"
    if OBP == "":
        OBP = x["OBP"].mean()
    else:
        try:
            OBP = float(OBP)
            df_predict["OBP"] = OBP
        except ValueError:
            return "Invalid Input for OBP"
    if SLG == "":
        SLG = x["SLG"].mean()
    else:
        try:
            SLG = float(SLG)
            df_predict["SLG"] = SLG
        except ValueError:
            return "Invalid Input for SLG"
    if OPS == "":
        OPS = x["OPS"].mean()
    else:
        try:
            OPS = float(OPS)
            df_predict["OPS"] = OPS
        except ValueError:
            return "Invalid Input for OPS"

    pickle_in = open("baseballwinpredictionmodel.pickle", "rb")
    model = pickle.load(pickle_in)
    result = model.predict(df_predict)

    if int(result) > 162:
        result = 162

    if int(result) < 0:
        result = 0

    result = "Predicted Wins: " + str(int(result))
    return result


def predict_outcome(month_, day_, start_time_, weather_, temp_, wind_, wind_speed_, attendance_, elapsed_time_):

    url_ = "https://raw.githubusercontent.com/devindhaliwal/baseball-analytics/master/Baseball-Analytica/games.csv"
    game_data = pd.read_csv(url_, sep=",")
    game_data[['wind_speed', 'wind_direction']] = game_data.wind.str.split(",", expand=True)
    game_data[['temp', 'weather']] = game_data.weather.str.split(",", expand=True)

    new_dates = []
    for date in game_data.date:
        mydate = datetime.datetime.strptime(date, '%Y-%m-%d')
        data = mydate.strftime('%B, %A')
        new_dates.append(data)

    game_data['new_dates'] = new_dates
    game_data[['month', 'day']] = game_data.new_dates.str.split(",", expand=True)
    game_data[['temp', 'deg']] = game_data.temp.str.split(" ", expand=True)
    game_data['temp'] = pd.to_numeric(game_data['temp'])
    game_data[['wind_speed', 'mph']] = game_data.wind_speed.str.split(" ", expand=True)
    game_data['wind_speed'] = pd.to_numeric(game_data['wind_speed'])
    game_data[['start_time', 'ampm']] = game_data.start_time.str.split(" ", expand=True)
    game_data[['hour', 'min']] = game_data.start_time.str.split(":", expand=True)
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

    game_data = game_data.filter(
        items=['month', 'day', 'outcome', 'away_final_score', 'home_final_score', 'elapsed_time', 'attendance',
               'start_time', 'wind_speed', 'wind_direction', 'temp', 'weather'])
    #game_data.to_csv("game_outcomes.csv")

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

    game_data = game_data.drop(
        ['month', 'day', 'wind_direction', 'weather', 'start_time', 'temp', 'wind_speed', 'elapsed_time', 'attendance',
         'home_final_score', 'away_final_score'], axis=1)

    game_data = game_data.join(oh_months)
    game_data = game_data.join(oh_days)
    game_data = game_data.join(oh_wind_direction)
    game_data = game_data.join(oh_weather)
    game_data = game_data.join(oh_start_time)
    game_data = game_data.join(oh_elapsed_time)
    game_data = game_data.join(oh_wind_speed)
    game_data = game_data.join(oh_temp)
    game_data = game_data.join(oh_attendance)

    game_data = game_data.drop(['outcome'], 1)
    df_predict = pd.DataFrame([game_data.mean() * 0])

    df_predict[month_] = 1
    df_predict[day_] = 1
    df_predict[start_time_] = 1
    df_predict[weather_] = 1
    df_predict[temp_] = 1
    df_predict[wind_] = 1
    df_predict[wind_speed_] = 1
    df_predict[attendance_] = 1
    df_predict[elapsed_time_] = 1

    pickle_in = open("baseballoutcomepredictionmodel.pickle", "rb")
    model = pickle.load(pickle_in)
    outcome = model.predict(df_predict)

    if outcome == 1:
        result = "Home Team Wins"
    else:
        result = "Away Team Wins"

    return result

def visualize_gameday_factors(x, y, hue):

    url_ = "https://raw.githubusercontent.com/devindhaliwal/baseball-analytics/master/Baseball-Analytica/games.csv"
    game_data = pd.read_csv(url_, sep=",")
    game_data[['wind_speed', 'wind_direction']] = game_data.wind.str.split(",", expand=True)
    game_data[['temp', 'weather']] = game_data.weather.str.split(",", expand=True)

    new_dates = []
    for date in game_data.date:
        mydate = datetime.datetime.strptime(date, '%Y-%m-%d')
        data_ = mydate.strftime('%B, %A')
        new_dates.append(data_)

    attendance = []
    for i in game_data.attendance:
        i = round(i / 5000) * 5000
        attendance.append(i)
    game_data["attendance"] = attendance

    elapsed_time = []
    for i in game_data.elapsed_time:
        i = round(i / 10) * 10
        elapsed_time.append(i)
    game_data["elapsed_time"] = elapsed_time

    game_data['new_dates'] = new_dates
    game_data[['month', 'day']] = game_data.new_dates.str.split(",", expand=True)
    game_data[['temp', 'deg']] = game_data.temp.str.split(" ", expand=True)
    game_data['temp'] = pd.to_numeric(game_data['temp'])
    game_data[['wind_speed', 'mph']] = game_data.wind_speed.str.split(" ", expand=True)
    game_data['wind_speed'] = pd.to_numeric(game_data['wind_speed'])
    game_data[['start_time', 'ampm']] = game_data.start_time.str.split(" ", expand=True)
    game_data[['hour', 'min']] = game_data.start_time.str.split(":", expand=True)
    game_data['hour'] = pd.to_numeric(game_data['hour'])
    game_data['min'] = pd.to_numeric(game_data['min'])

    game_data['start_time'] = game_data['hour']

    game_data = game_data.filter(items=['month','day','outcome','away_final_score','home_final_score','elapsed_time', 'attendance', 'start_time','wind_speed','wind_direction','temp','weather'])
    #game_data.to_csv("gameday_factors.csv")

    start_time_order = [11,12,1,2,3,4,5,6,7,8,9,10]
    month_order = ["March","April","May","June","July","August","September","October"]
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    if x == "month":
        order = month_order
    elif x == "day":
        order = day_order
    elif x == "start_time":
        order = start_time_order
    else:
        order = None

    sns.set()
    fig = plt.figure(figsize=(17, 10))
    sns.barplot(x=x, y=y, data=game_data, order=order, hue=hue)
    new_graph_name_im = "gameday_factors_graph_" + str(time.time()) + ".png"
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    image = img_data.read()
    s3 = boto3.resource('s3')
    s3.Object("baseball-analytica-graphs", new_graph_name_im).put(ACL='public-read', Body=bytes(image))
    link = "https://baseball-analytica-graphs.s3-us-west-1.amazonaws.com/" + new_graph_name_im
    return link
