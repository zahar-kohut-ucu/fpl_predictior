import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import random

BATCH_SIZE = 256
EPOCHS = 200
WINDOW_SIZE = 7
LR = 1e-3


cleaned_players = pd.read_csv("./clean_data/cleaned_merged_seasons.csv")
common_features = ['season_x', 'name', 'position', 'assists',
                   'creativity', 'goals_conceded',
                   'goals_scored', 'ict_index', 'influence', 'minutes',
                   'round', 
                   'total_points', 'value', 'selected']

train_features = ['assists',
                   'creativity', 'goals_conceded',
                   'goals_scored', 'ict_index', 'influence', 'minutes', 
                   'total_points', 'value', 'selected']

cleaned_players = cleaned_players[common_features]
cleaned_players = cleaned_players[cleaned_players['position'] != 'GK']
train_data = cleaned_players[cleaned_players["season_x"] != "2023-24"]
test_data = cleaned_players[cleaned_players["season_x"] == "2023-24"]

train_data = train_data.sort_values(by=['season_x', 'name', 'round']).reset_index(drop=True)

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
class FPLPredictor(nn.Module):
    def __init__(self, num_channels):
        super(FPLPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=32, num_layers=1, batch_first=True)  
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        output = self.fc(lstm_out[:, -1, :])  
        return output

#GRU
# class FPLPredictor(nn.Module):
#     def __init__(self, num_channels):
#         super(FPLPredictor, self).__init__()
#         self.gru = nn.GRU(input_size=num_channels, hidden_size=32, num_layers=1, batch_first=True)  
#         self.fc = nn.Linear(32, 1)
    
#     def forward(self, x):
#         gru_out, _ = self.gru(x)  
#         output = self.fc(gru_out[:, -1, :])  
#         return output

#TCN, need revision
# class FPLPredictor(nn.Module):
#     def __init__(self, num_channels, num_filters = 256, kernel_size=2, num_layers=2):
#         super(FPLPredictor, self).__init__()
        
#         self.encoder = nn.ModuleList()
#         for _ in range(num_layers):
#             self.encoder.append(
#                 nn.Conv1d(in_channels=num_channels if _ == 0 else num_filters, 
#                         out_channels=num_filters, 
#                         kernel_size=kernel_size, 
#                         padding='same')  
#             )
#             self.encoder.append(nn.MaxPool1d(kernel_size=2, stride=2))  
        
#         self.decoder = nn.ModuleList()
#         for _ in range(num_layers):
#             self.decoder.append(
#                 nn.Conv1d(in_channels=num_filters, 
#                                 out_channels=num_filters, 
#                                 kernel_size=kernel_size, 
#                                 padding='same')  
#             )
#             self.decoder.append(nn.Upsample(scale_factor=2, mode='nearest'))  
#         self.fc = nn.Linear(num_filters, 1)

        
#     def forward(self, x):
#         for layer in self.encoder:
#             if isinstance(layer, nn.Conv1d):
#                 x = torch.relu(layer(x))
#             else:
#                 x = layer(x)  
#         for layer in self.decoder:
#             if isinstance(layer, nn.ConvTranspose1d):
#                 x = torch.relu(layer(x))
#             else:
#                 x = layer(x)  

#         x = self.fc(x[:, :, -1])
#         return x

train_data["value_unscaled"] = train_data["value"]
scaler = RobustScaler()
train_data[train_features] = scaler.fit_transform(train_data[train_features])
X, y = [], []
for (season, player), player_data in train_data.groupby(['season_x', 'name']):
    player_data = player_data.reset_index(drop=True)
        
    if len(player_data) < WINDOW_SIZE:
        continue

    for i in range(len(player_data) - WINDOW_SIZE):
        X.append(player_data[train_features].iloc[i:i+WINDOW_SIZE].values)
        y.append(player_data['value_unscaled'].iloc[i + WINDOW_SIZE])
            
if not X:
    print(f"No sufficient data for position: {position}")  

X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
y = torch.tensor(np.array(y), dtype=torch.float32).to(device)

X = X.reshape((X.shape[0], X.shape[1], len(train_features)))#.transpose(1,2)
model = FPLPredictor(num_channels=len(train_features)).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss() 

dataset = torch.utils.data.TensorDataset(X, y)
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

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


# files = glob.glob('gws/gw*.csv')
# print(files)
# test_data = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
# test_data = test_data[list(set(common_features).intersection(set(test_data.columns)))]


test_data = test_data.sort_values(by=['name']).reset_index(drop=True) 
test_data["value_unscaled"] = test_data["value"]
test_data[train_features] = scaler.fit_transform(test_data[train_features])
test_X, test_y = [], []
for player, player_data in test_data.groupby(['name']):
    player_data = player_data.reset_index(drop=True)
    if len(player_data) < WINDOW_SIZE:
        continue

    for i in range(len(player_data) - WINDOW_SIZE):
        test_X.append(player_data[train_features].iloc[i:i+WINDOW_SIZE].values)
        test_y.append(player_data['value_unscaled'].iloc[i + WINDOW_SIZE])
            
if not test_X:
    print(f"No sufficient data for position: {position}")
    


test_X = torch.tensor(np.array(test_X), dtype=torch.float32).to(device)
test_y = torch.tensor(np.array(test_y), dtype=torch.float32).to(device)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], len(train_features)))#.transpose(1,2)

test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("------------------")
print("Evaluate on normalized test data: ")
model.eval()  
test_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        test_loss += criterion(y_pred.squeeze(), y_batch).item()

test_loss /= len(test_loader)
print(f"Test Loss (MAE): {test_loss}")

predictions = model(test_X).cpu().detach().numpy()
accuracy = np.mean(np.round(predictions) == np.round(test_y.cpu().detach().numpy())) * 100

indices = random.sample(range(len(test_X)), min(5, len(test_X)))
for idx in indices:
    print(f"Predicted Value: {predictions[idx]}")
    print(f"Actual Value: {test_y[idx]}\n")

print(f"Overall accuracy:", accuracy)
