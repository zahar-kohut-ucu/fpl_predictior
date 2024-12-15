import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import random

BATCH_SIZES = [128, 256, 256, 128]
EPOCHS = 100
WINDOW_SIZE = 7
LR = 1e-3
SCALED = True

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

print(device)

def print_gpu_usage():
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated()
        total_memory = torch.cuda.memory_reserved()
        print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")
        print(f"Total memory reserved: {total_memory / 1024**2:.2f} MB")
    else:
        print("CUDA is not available.")

# LSTM
# class FPLPredictor(nn.Module):
#     def __init__(self, num_channels):
#         super(FPLPredictor, self).__init__()
#         self.lstm = nn.LSTM(input_size=num_channels, hidden_size=32, num_layers=1, batch_first=True)  
#         self.fc = nn.Linear(32, 1)
    
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)  
#         output = self.fc(lstm_out[:, -1, :])  # Use the output of the last time step
#         return output

# GRU
# class FPLPredictor(nn.Module):
#     def __init__(self, num_channels):
#         super(FPLPredictor, self).__init__()
#         self.lstm = nn.GRU(input_size=num_channels, hidden_size=32, num_layers=1, batch_first=True)  
#         self.fc = nn.Linear(32, 1)
    
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)  
#         output = self.fc(lstm_out[:, -1, :])  # Use the output of the last time step
#         return output

#TCN
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
        self.fc = nn.Linear(num_filters, 1)

        
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
        
        if len(player_data) < WINDOW_SIZE:
            continue

        for i in range(len(player_data) - WINDOW_SIZE):
            X.append(player_data[features].iloc[i:i+WINDOW_SIZE].values)
            if SCALED:
                y.append(player_data['value'].iloc[i + WINDOW_SIZE])
            else:
                y.append(player_data['value_unscaled'].iloc[i + WINDOW_SIZE])
            
    if not X:
        print(f"No sufficient data for position: {position}")   
        continue

    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(y), dtype=torch.float32).to(device)
    
    X = X.reshape((X.shape[0], X.shape[1], len(features))).transpose(1,2)
    model = FPLPredictor(num_channels=len(features)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss() 

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
            loss = criterion(y_pred.squeeze(), y_batch)  
            loss.backward()  
            optimizer.step()  
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                #X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred_val = model(X_val)
                val_loss += criterion(y_pred_val.squeeze(), y_val).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")
    p += 1
    models[position] = model

files = glob.glob('gws/gw*.csv')
test_data = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
test_data = test_data[list(set(common_features).intersection(set(test_data.columns)))]

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
        
        if len(player_data) < WINDOW_SIZE:
            continue

        for i in range(len(player_data) - WINDOW_SIZE):
            test_X.append(player_data[features].iloc[i:i+WINDOW_SIZE].values)
            if SCALED:
                test_y.append(player_data['value'].iloc[i + WINDOW_SIZE])
            else:
                test_y.append(player_data['value_unscaled'].iloc[i + WINDOW_SIZE])
            
    if not test_X:
        print(f"No sufficient data for position: {position}")
        continue

    model = models[position]

    test_X = torch.tensor(np.array(test_X), dtype=torch.float32).to(device)
    test_y = torch.tensor(np.array(test_y), dtype=torch.float32).to(device)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], len(features))).transpose(1,2)

    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZES[p])

    print("------------------")
    print("Evaluate on normalized test data: ")
    model.eval()  
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            test_loss += criterion(y_pred.squeeze(), y_batch).item()

    test_loss /= len(test_loader)
    print(f"Test Loss (MAE) for {position}: {test_loss}")
    p += 1
    predictions = model(test_X).cpu().detach().numpy()

    if SCALED:
        num_features = len(features)  
        dummy_predictions = np.zeros((len(predictions), num_features))
        dummy_predictions[:, -1] = predictions[:, 0]  
        unnormalized_predictions = scaler.inverse_transform(dummy_predictions)[:, -1]  

        dummy_test_y = np.zeros((len(test_y), num_features))
        dummy_test_y[:, -1] = test_y.cpu().detach().numpy()
        unnormalized_test_y = scaler.inverse_transform(dummy_test_y)[:, -1]

        accuracy = np.mean(np.round(unnormalized_predictions) == np.round(unnormalized_test_y)) * 100
    else:
        accuracy = np.mean(np.round(predictions) == np.round(test_y.cpu().detach().numpy())) * 100


    indices = random.sample(range(len(test_X)), min(5, len(test_X)))
    for idx in indices:
        print(f"Position: {position}")
        if SCALED:
            print(f"Predicted Value (unnormalized): {unnormalized_predictions[idx]:.2f}")
            print(f"Actual Value (unnormalized): {unnormalized_test_y[idx]:.2f}\n")
        else:
            print(f"Predicted Value (unnormalized): {predictions[idx]}")
            print(f"Actual Value (unnormalized): {test_y.cpu().detach().numpy()[idx]}\n")
    
    print(f"Overall accuracy for {position}:", accuracy)
