import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("games.csv")
df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])
df = df.sort_values("GAME_DATE_EST").reset_index(drop=True)

home = df[["GAME_DATE_EST", "HOME_TEAM_ID", "PTS_home", "FG_PCT_home", "REB_home", "AST_home", "HOME_TEAM_WINS"]].copy()
home.columns = ["date", "team_id", "pts", "fg_pct", "reb", "ast", "win"]
home["is_home"] = 1

away = df[["GAME_DATE_EST", "VISITOR_TEAM_ID", "PTS_away", "FG_PCT_away", "REB_away", "AST_away", "HOME_TEAM_WINS"]].copy()
away.columns = ["date", "team_id", "pts", "fg_pct", "reb", "ast", "win"]
away["win"] = 1 - away["win"]
away["is_home"] = 0

team_games = pd.concat([home, away]).sort_values(["team_id", "date"]).reset_index(drop=True)

roll_cols = ["pts", "fg_pct", "reb", "ast", "win"]
for col in roll_cols:
    team_games[f"roll_{col}"] = (
        team_games.groupby("team_id")[col]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    )

team_games["rest_days"] = team_games.groupby("team_id")["date"].diff().dt.days

# Elo ratings: self-correcting team strength, updated after every game
elo = {}
K = 20
HOME_ADV = 100
pre_game_elo = []

games_sorted = df.sort_values("GAME_DATE_EST")
for _, g in games_sorted.iterrows():
    h_id, a_id = g["HOME_TEAM_ID"], g["VISITOR_TEAM_ID"]
    h_elo = elo.get(h_id, 1500)
    a_elo = elo.get(a_id, 1500)
    pre_game_elo.append({"GAME_ID": g["GAME_ID"], "home_elo": h_elo, "away_elo": a_elo})

    expected_home = 1 / (1 + 10 ** (((a_elo) - (h_elo + HOME_ADV)) / 400))
    actual_home = g["HOME_TEAM_WINS"]
    elo[h_id] = h_elo + K * (actual_home - expected_home)
    elo[a_id] = a_elo + K * ((1 - actual_home) - (1 - expected_home))

elo_df = pd.DataFrame(pre_game_elo)

roll_features = [f"roll_{c}" for c in roll_cols] + ["rest_days"]
team_feats = team_games[["team_id", "date"] + roll_features].drop_duplicates(subset=["team_id", "date"])

home_feats = team_feats.add_prefix("home_").rename(columns={"home_team_id": "HOME_TEAM_ID", "home_date": "GAME_DATE_EST"})
away_feats = team_feats.add_prefix("away_").rename(columns={"away_team_id": "VISITOR_TEAM_ID", "away_date": "GAME_DATE_EST"})

model_df = df.merge(home_feats, on=["HOME_TEAM_ID", "GAME_DATE_EST"], how="left")
model_df = model_df.merge(away_feats, on=["VISITOR_TEAM_ID", "GAME_DATE_EST"], how="left")
model_df = model_df.merge(elo_df, on="GAME_ID", how="left")
model_df["elo_diff"] = model_df["home_elo"] - model_df["away_elo"]

feature_cols = [c for c in model_df.columns if c.startswith("home_roll") or c.startswith("home_rest") or c.startswith("away_roll") or c.startswith("away_rest")]
feature_cols += ["home_elo", "away_elo", "elo_diff"]
model_df = model_df[feature_cols + ["HOME_TEAM_WINS"]].dropna().rename(columns={"HOME_TEAM_WINS": "target"})
split_idx = int(len(model_df) * 0.8)
train, test = model_df.iloc[:split_idx], model_df.iloc[split_idx:]

scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols])
X_test = scaler.transform(test[feature_cols])
y_train, y_test = train["target"], test["target"]

print("Class balance (target):")
print(model_df["target"].value_counts(normalize=True))

baseline_acc = max(y_test.mean(), 1 - y_test.mean())
print(f"\nNaive baseline (always guess majority class) — Accuracy: {baseline_acc:.3f}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} — Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))