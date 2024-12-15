import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import random

BATCH_SIZES = [128, 256, 256, 128]
EPOCHS = 15
WINDOW_SIZE = 8
PRED_SIZE = 3
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
train_data = cleaned_players[cleaned_players["season_x"] != "2023-24"]
test_data = cleaned_players[cleaned_players["season_x"] == "2023-24"]

ds_dict = {
    'GK': (train_data[train_data['position'] == 'GK'], gk_features),
    'DEF': (train_data[train_data['position'] == 'DEF'], defender_features),
    'MID': (train_data[train_data['position'] == 'MID'], mid_features),
    'FWD': (train_data[train_data['position'] == 'FWD'], fwd_features)
}

scalers = {}
models = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#print(device)

def print_gpu_usage():
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated()
        total_memory = torch.cuda.memory_reserved()
        print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")
        print(f"Total memory reserved: {total_memory / 1024**2:.2f} MB")
    else:
        print("CUDA is not available.")

#LSTM
# class FPLPredictor(nn.Module):
#     def __init__(self, num_channels):
#         super(FPLPredictor, self).__init__()
#         self.lstm = nn.LSTM(input_size=num_channels, hidden_size=32, num_layers=1, batch_first=True)  
#         self.fc = nn.Linear(32, 3)
    
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)  
#         output = self.fc(lstm_out[:, -1, :])  
#         output = torch.nn.functional.softmax(output, dim=-1)
#         return output

#GRU
# class FPLPredictor(nn.Module):
#     def __init__(self, num_channels):
#         super(FPLPredictor, self).__init__()
#         self.gru = nn.GRU(input_size=num_channels, hidden_size=32, num_layers=1, batch_first=True)  
#         self.fc = nn.Linear(32, 5)
    
#     def forward(self, x):
#         gru_out, _ = self.gru(x)  
#         output = self.fc(gru_out[:, -1, :])  
#         output = torch.nn.functional.softmax(output, dim=-1)
#         return output

#TCN, need revision
class FPLPredictor(nn.Module):
    def __init__(self, num_channels, num_filters = 256, kernel_size=2, num_layers=2):
        super(FPLPredictor, self).__init__()
        
        self.encoder = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder.append(
                nn.Conv1d(in_channels=num_channels if _ == 0 else num_filters, 
                        out_channels=num_filters, 
                        kernel_size=kernel_size, 
                        padding='same')  
            )
            self.encoder.append(nn.MaxPool1d(kernel_size=2, stride=2))  
        
        self.decoder = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder.append(
                nn.Conv1d(in_channels=num_filters, 
                                out_channels=num_filters, 
                                kernel_size=kernel_size, 
                                padding='same')  
            )
            self.decoder.append(nn.Upsample(scale_factor=2, mode='nearest'))  
        self.fc = nn.Linear(num_filters, 3)

        
    def forward(self, x):
        for layer in self.encoder:
            if isinstance(layer, nn.Conv1d):
                x = torch.relu(layer(x))
            else:
                x = layer(x)  
        for layer in self.decoder:
            if isinstance(layer, nn.ConvTranspose1d):
                x = torch.relu(layer(x))
            else:
                x = layer(x)  

        x = self.fc(x[:, :, -1])
        x = torch.nn.functional.softmax(x, dim=-1)
        return x

p = 0
for position, (position_data, features) in ds_dict.items():
    position_data = position_data.sort_values(by=['season_x', 'name', 'round']).reset_index(drop=True)
    
    scaler = RobustScaler()
    scalers[position] = scaler 
    position_data["value_unscaled"] = position_data["value"]
    position_data[features] = scaler.fit_transform(position_data[features])
    X, y = [], []
    for (season, player), player_data in position_data.groupby(['season_x', 'name']):
        player_data = player_data.reset_index(drop=True)
        
        if len(player_data) < WINDOW_SIZE + PRED_SIZE:
            continue

        for i in range(len(player_data) - WINDOW_SIZE - PRED_SIZE):
            X.append(player_data[features].iloc[i:i+WINDOW_SIZE].values)
            current_value = player_data['value_unscaled'].iloc[i + WINDOW_SIZE]
            future_values = player_data['value_unscaled'].iloc[i + WINDOW_SIZE + 1:i + WINDOW_SIZE + PRED_SIZE + 1]
            value_change = future_values - current_value
            
            max_change = value_change.max()
            min_change = value_change.min()

            if max_change >= 2:
                y.append(4)  
            elif max_change >= 1:
                y.append(3)  
            elif min_change <= -2:
                y.append(0)  
            elif min_change <= -1:
                y.append(1) 
            else:
                y.append(2)  
            
    if not X:
        print(f"No sufficient data for position: {position}")   
        continue

    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(y), dtype=torch.long).to(device)

    X = X.reshape((X.shape[0], X.shape[1], len(features)))#.transpose(1,2)
    y = torch.nn.functional.one_hot(y, num_classes=5).float().to(device)

    model = FPLPredictor(num_channels=len(features)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZES[p], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZES[p])

    models[position] = model
    for epoch in range(EPOCHS):
        #print_gpu_usage()
        model.train()
        for X_batch, y_batch in train_loader:
            #X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()  
            y_pred = model(X_batch)  
            #print(y_pred)
            loss = criterion(y_pred, y_batch)  
            loss.backward()  
            optimizer.step()  
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                #X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred_val = model(X_val)
                val_loss += criterion(y_pred_val, y_val).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")
    p += 1
    models[position] = model

# files = glob.glob('gws/gw*.csv')
# test_data = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
# test_data = test_data[list(set(common_features).intersection(set(test_data.columns)))]

test_ds_dict = {
    'GK': (test_data[test_data['position'] == 'GK'], gk_features),
    'DEF': (test_data[test_data['position'] == 'DEF'], defender_features),
    'MID': (test_data[test_data['position'] == 'MID'], mid_features),
    'FWD': (test_data[test_data['position'] == 'FWD'], fwd_features)
}

p = 0
for position, (position_data, features) in test_ds_dict.items():
    position_data = position_data.sort_values(by=['name']).reset_index(drop=True)
    scaler = scalers[position]  
    position_data["value_unscaled"] = position_data["value"]
    position_data[features] = scaler.fit_transform(position_data[features])
    test_X, test_y = [], []
    for player, player_data in position_data.groupby(['name']):
        player_data = player_data.reset_index(drop=True)
        
        if len(player_data) < WINDOW_SIZE + PRED_SIZE:
            continue

        for i in range(len(player_data) - WINDOW_SIZE - PRED_SIZE):
            test_X.append(player_data[features].iloc[i:i+WINDOW_SIZE].values)
            current_value = player_data['value_unscaled'].iloc[i + WINDOW_SIZE]
            future_values = player_data['value_unscaled'].iloc[i + WINDOW_SIZE + 1:i + WINDOW_SIZE + PRED_SIZE + 1]
            value_change = future_values - current_value
            max_change = value_change.max()
            min_change = value_change.min()

            if max_change >= 2:
                test_y.append(4)  
            elif max_change >= 1:
                test_y.append(3)  
            elif min_change <= -2:
                test_y.append(0)  
            elif min_change <= -1:
                test_y.append(1) 
            else:
                test_y.append(2)  
            
    if not test_X:
        print(f"No sufficient data for position: {position}")
        continue

    model = models[position]

    test_X = torch.tensor(np.array(test_X), dtype=torch.float32).to(device)
    test_y = torch.tensor(np.array(test_y), dtype=torch.long).to(device)

    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], len(features)))#.transpose(1,2)
    test_y = torch.nn.functional.one_hot(test_y, num_classes=5).float().to(device)

    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZES[p])

    print("------------------")
    print("Evaluate on normalized test data: ")
    model.eval()  
    correct_test = 0
    total_test = 0
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            test_loss += criterion(y_pred, y_batch).item()
            _, predicted = torch.max(y_pred, 1)
            _, labels = torch.max(y_batch, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    print(f"Test Loss (Cross-Correlation) for {position}: {test_loss}")
    print(f"Test Accuracy for {position}: {test_accuracy}%")
    p += 1

