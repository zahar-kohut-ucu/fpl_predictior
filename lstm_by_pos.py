import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam
import glob
import random

BATCH_SIZE = 128
EPOCHS = 50
WINDOW_SIZE = 7
LR = 1e-3

gk_features = ['goals_conceded', 'influence', 'minutes',
                'own_goals', 'saves', 'selected',
                'total_points', 'value']
defender_features = ['assists', 'ict_index', 'goals_conceded',
                        'goals_scored', 'influence', 'minutes', 
                        'selected', 'threat', 
                        'total_points', 'value']
mid_features = ['assists', 'creativity', 'ict_index', 'goals_scored', 
                'influence', 'minutes', 'selected', 
                'threat', 'total_points', 'value']
fwd_features = ['assists', 'creativity', 'ict_index', 'goals_scored', 
                'influence', 'minutes', 'selected', 'threat', 
                'total_points', 'value']

cleaned_players = pd.read_csv("./clean_data/cleaned_merged_seasons.csv")
common_features = ['season_x', 'name', 'position', 'assists',
                   'clean_sheets', 'creativity', 'goals_conceded',
                   'goals_scored', 'ict_index', 'influence', 'minutes',
                   'own_goals', 'penalties_saved', 'red_cards', 'round', 
                   'saves', 'selected', 'threat', 'total_points', 'value', 
                   'yellow_cards']
cleaned_players = cleaned_players[common_features]

ds_dict = {
    'GK': (cleaned_players[cleaned_players['position'] == 'GK'], gk_features),
    'DEF': (cleaned_players[cleaned_players['position'] == 'DEF'], defender_features),
    'MID': (cleaned_players[cleaned_players['position'] == 'MID'], mid_features),
    'FWD': (cleaned_players[cleaned_players['position'] == 'FWD'], fwd_features)
}

scalers = {}
models = {}

for position, (position_data, features) in ds_dict.items():
    position_data = position_data.sort_values(by=['season_x', 'name', 'round']).reset_index(drop=True)
    
    scaler = RobustScaler()
    scalers[position] = scaler 
    position_data[features] = scaler.fit_transform(position_data[features])
    X, y = [], []
    for (season, player), player_data in position_data.groupby(['season_x', 'name']):
        player_data = player_data.reset_index(drop=True)
        
        if len(player_data) < WINDOW_SIZE:
            continue

        for i in range(len(player_data) - WINDOW_SIZE):
            X.append(player_data[features].iloc[i:i+WINDOW_SIZE].values)
            y.append(player_data['value'].iloc[i + WINDOW_SIZE])
            
    if not X:
        print(f"No sufficient data for position: {position}")
        continue

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], len(features)))

    model = Sequential()
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=LR), loss='mae')
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.05)
    models[position] = model


files = glob.glob('gws/gw*.csv')
print(files)
test_data = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
test_data = test_data[list(set(common_features).intersection(set(test_data.columns)))]

test_ds_dict = {
    'GK': (test_data[test_data['position'] == 'GK'], gk_features),
    'DEF': (test_data[test_data['position'] == 'DEF'], defender_features),
    'MID': (test_data[test_data['position'] == 'MID'], mid_features),
    'FWD': (test_data[test_data['position'] == 'FWD'], fwd_features)
}


for position, (position_data, features) in test_ds_dict.items():
    position_data = position_data.sort_values(by=['name']).reset_index(drop=True)
    scaler = scalers[position]  
    position_data[features] = scaler.fit_transform(position_data[features])
    test_X, test_y = [], []
    for player, player_data in position_data.groupby(['name']):
        player_data = player_data.reset_index(drop=True)
        
        if len(player_data) < WINDOW_SIZE:
            continue

        for i in range(len(player_data) - WINDOW_SIZE):
            test_X.append(player_data[features].iloc[i:i+WINDOW_SIZE].values)
            test_y.append(player_data['value'].iloc[i + WINDOW_SIZE])
            
    if not test_X:
        print(f"No sufficient data for position: {position}")
        continue

    test_X = np.array(test_X)
    test_y = np.array(test_y)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], len(features)))
    
    print("------------------")
    print("Evaluate on normalized test data: ")
    results = models[position].evaluate(test_X, test_y, batch_size=BATCH_SIZE)
    print("normalized test loss:", results)

    predictions = models[position].predict(test_X)

    num_features = len(features)  
    dummy_predictions = np.zeros((len(predictions), num_features))
    dummy_predictions[:, -1] = predictions[:, 0]  
    unnormalized_predictions = scaler.inverse_transform(dummy_predictions)[:, -1]  

    dummy_test_y = np.zeros((len(test_y), num_features))
    dummy_test_y[:, -1] = test_y
    unnormalized_test_y = scaler.inverse_transform(dummy_test_y)[:, -1]

    accuracy = np.mean(np.round(unnormalized_predictions) == np.round(unnormalized_test_y)) * 100

    indices = random.sample(range(len(test_X)), min(5, len(test_X)))
    for idx in indices:
        print(f"Position: {position}")
        print(f"Predicted Value (unnormalized): {unnormalized_predictions[idx]:.2f}")
        print(f"Actual Value (unnormalized): {unnormalized_test_y[idx]:.2f}\n")

    print(f"Evaluate on test data for {position}: ")
    results = models[position].evaluate(test_X, test_y, batch_size=BATCH_SIZE)
    print(f"test loss:", results)
    
    print(f"Overall accuracy for {position}:", accuracy)
