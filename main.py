
#***************************************************************************************************
#                                           IMPORTING LIBRARIES
#***************************************************************************************************

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.modules import MSELoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pkbar
from torch.autograd import Variable

#****************************************************************************************************
#                                            PARAMETERS
#****************************************************************************************************

batch_size = 128
epochs = 2
learning_rate = 0.1
split_ratio = 0.2

#****************************************************************************************************
#                                       
#****************************************************************************************************

# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 0)

class Dataset_Generator(Dataset):
     # Constructor Function
    def __init__(self, x_data_path, y_data_path):
        self.df_x = pd.read_csv (x_data_path, header = 0)
        self.df_y = pd.read_csv (y_data_path, header = 0)
        self.x_data =  self.df_x.to_numpy()
        self.y_data =  self.df_y.to_numpy()
        print("Dataset Instance initialized")
        
    # Length for batch
    def __len__(self):
        return len(self.x_data)
        
    def __getitem__(self, index):
        tensor_x_data = torch.tensor(self.x_data[index,None,:])
        tensor_y_data = torch.tensor(self.y_data[index,None,:])
        return (tensor_x_data, tensor_y_data)


####################################################################################################
#                                         Network
####################################################################################################

class Autoencoder(nn.Module):
    
    # Constructor
    
    def __init__(self):
        
        # Encoder Part
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 40, 16, stride=2, padding = 7), 
            nn.BatchNorm1d(40, momentum=0.1),
            nn.ReLU(True),
            nn.Conv1d(40, 20, 16, stride=2, padding = 7),
            nn.BatchNorm1d(20, momentum=0.1),
            nn.ReLU(True),
            nn.Conv1d(20, 20, 16, stride=2, padding = 7), 
            nn.BatchNorm1d(20, momentum=0.1),
            nn.ReLU(True),
            nn.Conv1d(20, 20, 16, stride=2, padding = 7),
            nn.BatchNorm1d(20, momentum=0.1),
            nn.ReLU(True),
            nn.Conv1d(20, 40, 16, stride=2, padding = 7),
            nn.BatchNorm1d(40, momentum=0.1),
            nn.ReLU(True),
            nn.Conv1d(40, 1, 15, stride=1, padding = 7),
            nn.BatchNorm1d(1, momentum=0.1),
            nn.ReLU(True)
        )
        
        # Decoder Part
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 40, 15, stride=1, padding = 7),  # b, 16, 5, 5
            nn.BatchNorm1d(40, momentum=0.1),
            nn.ReLU(True),
            nn.ConvTranspose1d(40, 20, 16, stride=2, padding = 7),  # b, 16, 5, 5
            nn.BatchNorm1d(20, momentum=0.1),
            nn.ReLU(True),
            nn.ConvTranspose1d(20, 20, 16, stride=2, padding = 7),  # b, 8, 15, 15
            nn.BatchNorm1d(20, momentum=0.1),
            nn.ReLU(True),
            nn.ConvTranspose1d(20, 20, 16, stride=2,padding = 7),  # b, 1, 28, 28
            nn.BatchNorm1d(20, momentum=0.1),
            nn.ReLU(True),
            nn.ConvTranspose1d(20, 20, 16, stride=2, padding = 7),  # b, 1, 28, 28
            nn.BatchNorm1d(20, momentum=0.1),
            nn.ReLU(True),
            nn.ConvTranspose1d(20, 40, 16, stride=2, padding = 7),  # b, 1, 28, 28
            nn.BatchNorm1d(40, momentum=0.1),
            nn.ReLU(True),
            nn.ConvTranspose1d(40, 1, 17, stride=1, padding = 8)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Generator for Train and Test Data
dataset = Dataset_Generator('x_train.csv', 'y_train.csv')

# Loading the data into batches
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = False)


####################################################################################################
#                                   MAIN CODE
####################################################################################################



# Defining the Model
model = Autoencoder()
model.to(device)
print("Model loaded on GPU")

# Defining the Optimziers and Loss Functions
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
loss = MSELoss() 
train_per_epoch = int(dataset.__len__()/128)
target = int(train_per_epoch*(1-split_ratio))

# Training and Validation Loop
print("Starting the Training")
for k in range(epochs):
    
    # EPOCH START
    print("\nEpoch = ",k+1)
    kbar = pkbar.Kbar(target=target+1, width=12)
    
    # Training and Validation of Epoch
    j, final_val_loss = 0, torch.tensor(0)
    for i,(input_data, output_data) in enumerate(dataloader):
    
        # Loading data on GPU
        input_data = Variable(input_data.float().to(device))
        output_data = Variable(output_data.float().to(device))

        # TRAINING THE MODEL
        if i<= int(train_per_epoch*(1-split_ratio)):
            
            # Forward Feed Data
            predicted_data = model.forward(input_data)
            training_loss = loss(output_data,predicted_data)
            
            # Backward Propogation and weights update
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
            kbar.update(i, values=[("Training_loss", training_loss)])

        # VALIDATION OF EPOCH
        else: 
            val_predicted_output = model(input_data.float())
            val_loss = loss(val_predicted_output, output_data)
            j = j+1
        
        # EPOCH END
            
    kbar.update(1, values=[("training Loss", training_loss), ("Val_Loss", val_loss)])
