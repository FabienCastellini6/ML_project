#Packages
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

train_home_team_statistics_df = pd.read_csv('C:/Users/FABIEN/Documents/Dauphine/Machine Learning/Train_Data/train_home_team_statistics_df.csv', index_col=0)
train_away_team_statistics_df = pd.read_csv('C:/Users/FABIEN/Documents/Dauphine/Machine Learning/Train_Data/train_away_team_statistics_df.csv', index_col=0)

train_scores = pd.read_csv('C:/Users/FABIEN/Documents/Dauphine/Machine Learning/Y_train.csv', index_col=0)

#Données joueurs
train_home_players_statistics_df = pd.read_csv('C:/Users/FABIEN/Documents/Dauphine/Machine Learning/Train_Data/train_home_player_statistics_df.csv', index_col=0)
train_away_players_statistics_df = pd.read_csv('C:/Users/FABIEN/Documents/Dauphine/Machine Learning/Train_Data/train_away_player_statistics_df.csv', index_col=0)

home_wins = train_scores[train_scores["HOME_WINS"]==1]
        
home_teams = train_home_team_statistics_df[train_home_team_statistics_df.index.isin(home_wins.index)]
away_teams = train_away_team_statistics_df[train_away_team_statistics_df.index.isin(home_wins.index)]

#Modèle
results = []
for colonne in home_teams.columns[2:]:
    shoots = pd.DataFrame()
    shoots['home'] = home_teams[colonne]
    shoots['away'] = away_teams[colonne]
    shoots['diff'] = shoots['home'] - shoots['away']
    shoots['predic_home'] = 0
    shoots['predic_draw'] = 0
    shoots['predic_away'] = 0
    
    for index, row in shoots.iterrows():
        if row['diff'] > 0:
            shoots.at[index, 'predic_home'] = 1
        elif row['diff'] == 0:
            shoots.at[index, 'predic_draw'] = 1
        else:
            shoots.at[index, 'predic_away'] = 1
            
        
    predictions_2 = shoots.iloc[:, -3:]

    results.append(np.round(accuracy_score(predictions_2,home_wins),4).tolist())
    print(np.round(accuracy_score(predictions_2,home_wins),4))
    
nouveau_df = train_home_team_statistics_df.columns.to_frame().stack().reset_index(drop=True).to_frame(name='Nom de la colonne')
nouveau_df = nouveau_df.iloc[2:]
nouveau_df['results'] = results
