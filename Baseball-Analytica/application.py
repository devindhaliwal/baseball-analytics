from flask import Flask, request, render_template
import baseball_analytica

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/data/")
def data():
    return render_template("data.html")

@app.route("/data/predictwins/", methods=["GET","POST"])
def predict_wins():
    if request.method == "POST":
        AB = request.form["AB"]
        R = request.form["R"]
        H = request.form["H"]
        _2B = request.form["2B"]
        _3B = request.form["3B"]
        HR = request.form["HR"]
        BB = request.form["BB"]
        SO = request.form["SO"]
        SB = request.form["SB"]
        CS = request.form["CS"]
        RA = request.form["RA"]
        ER = request.form["ER"]
        ERA = request.form["ERA"]
        CG = request.form["CG"]
        SHO = request.form["SHO"]
        SV = request.form["SV"]
        HA = request.form["HA"]
        HRA = request.form["HRA"]
        BBA = request.form["BBA"]
        SOA = request.form["SOA"]
        E = request.form["E"]
        DP = request.form["DP"]
        FP = request.form["FP"]
        BA = request.form["BA"]
        OBP = request.form["OBP"]
        SLG = request.form["SLG"]
        OPS = request.form["OPS"]
        result = baseball_analytica.predict_wins(AB, R, H, _2B, _3B, HR, BB, SO,
                                    SB, CS, RA, ER, ERA, CG, SHO, SV, HA, HRA, 
                                    BBA, SOA, E, DP, FP, BA, OBP, SLG, OPS)
        return render_template("predict_wins.html", output=result)
    else:
        return render_template("predict_wins.html")

@app.route("/data/statvisualization", methods=["GET","POST"])
def stat_visualization():
    if request.method == "POST":
        stat = request.form["stat"]
        try:
            stat = float(stat)
            return render_template("stat_visualization.html", error="Invalid Input")
        except ValueError:
            stat = stat.upper()
            image = baseball_analytica.graph_stat(stat)
            if image == "Invalid Input":
                return render_template("stat_visualization.html", error="Invalid Input")
            return render_template("stat_visualization.html", plot=image)
    else:
        return render_template("stat_visualization.html")

@app.route("/data/teamstatvisualization", methods=["GET","POST"])
def team_stat_visualization():
    if request.method == "POST":
        stat = request.form["stat"]
        try:
            stat = float(stat)
            return render_template("team_stat_visualization.html", error="Invalid Input")
        except ValueError:
            stat = stat.upper()
            team = request.form["team"]
            try:
                team = float(team)
                return render_template("team_stat_visualization.html", error="Invalid Input")
            except ValueError:
                team = team.upper()
                try:
                    year = int(request.form["year"])
                    image = baseball_analytica.graph_team_stat(stat, team, year)
                    if image == "Invalid Input":
                        return render_template("team_stat_visualization.html", error="Invalid Input")
                    else:
                        return render_template("team_stat_visualization.html", plot=image)
                except ValueError:
                    return render_template("team_stat_visualization.html", error="Invalid Input")

    else:
        return render_template("team_stat_visualization.html")

@app.route("/data/predictoutcome", methods=["GET","POST"])
def predict_outcome():
    if request.method == "POST":
        month = request.form["month"]
        day = request.form["day"]
        start_time = request.form["start_time"]
        weather = request.form["weather"].lower()
        temp = request.form["temp"]
        wind = request.form["wind"]
        wind_speed = request.form["wind_speed"]
        attendance = request.form["attendance"]
        elapsed_time = request.form["elapsed_time"]
        result = baseball_analytica.predict_outcome(month, day, start_time, weather, temp, wind, wind_speed, attendance, elapsed_time)
        return render_template("predict_outcome.html", output=result)
    else:
        return render_template("predict_outcome.html")

@app.route("/data/gamedayfactorsvisualization", methods=["GET","POST"])
def gameday_factors_visualization():
    if request.method == "POST":
        x = request.form["x"]
        y = request.form["y"]
        hue = request.form["hue"]
        if hue == "None":
            hue = None
        image = baseball_analytica.visualize_gameday_factors(x, y, hue)
        return render_template("gameday_factors_visualization.html", plot=image)
    else:
        return render_template("gameday_factors_visualization.html")

@app.route("/articles/")
def articles():
    return render_template("articles.html")

@app.route("/articles/60-game-schedule-pitching-rules/")
def game_schedule_pitching_rules_60():
    return render_template("60-game-schedule-pitching-rules.html")

@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/resources/")
def resources():
    return render_template("resources.html")

if __name__ == "__main__":
    app.run(debug=False)