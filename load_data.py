
# Import the Libraries
import pandas as pd
import matplotlib.pyplot as plt
from pyecg import ECGRecord
import csv
import random
import requests
import zipfile


####################################################################################################
#                                     DOWNLOAD AND EXTRACT THE DATASET
###################################################################################################

noise_type = Input(" Enter the type of noise to be added (bw/em/ma):")

signal_dataset_url = "https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip"
noise_dataset_url =  "https://physionet.org/static/published-projects/nstdb/mit-bih-noise-stress-test-database-1.0.0.zip"

r = requests.get(image_url)   
with open("MIT_Signal.zip",'wb') as f: 
    f.write(r.content) 

r1 = requests.get(noise_dataset_url)
with open("MIT_Noise.zip", "wb") as f1:
    f1.write(r1.content)
 
with zipfile.ZipFile("MIT_Signal.zip","r") as zip_ref:
    zip_ref.extractall()

with zipfile.ZipFile("MIT_Noise.zip","r") as zip_ref:
    zip_ref.extractall()

hea_path_1 = "mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/"
hea_path = "mit-bih-noise-stress-test-database-1.0.0/mit-bih-noise-stress-test-database-1.0.0/bw.hea"



# To load a wfdb formatted ECG record
hea_path_1 = "D:/Research_Impl/Noise_Removal_Autoencoder/Dataset/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/"
hea_path = "D:/Research_Impl/Noise_Removal_Autoencoder/Dataset/mit-bih-noise-stress-test-database-1.0.0 (1)/mit-bih-noise-stress-test-database-1.0.0/bw.hea"



#######################################################################################################
#                                      EXTRACTING THE NOISE SIGNALS FROM HEA
#######################################################################################################

record = ECGRecord.from_wfdb(hea_path)
signal_noise = record.get_lead("noise1")

#######################################################################################################
#                                      EXTRACTING THE ORIGINAL SIGNALS FROM HEA
#######################################################################################################
signal_original = []
for i in range(100,125):
    if i == 110 or i == 120:
        continue
    record_1 = ECGRecord.from_wfdb(hea_path_1 + str(i) + ".hea")
    signal_orig = record_1.get_lead("MLII")
    if signal_orig == None:
        print(i)
        continue
    signal_original.append(signal_orig)


for i in range(200,235):
    if i == 204 or i == 206 or i == 211 or i == 216 or i == 218 or i == 224 or i == 225 or i == 226 or i == 227 or i == 229:
        continue
    record_1 = ECGRecord.from_wfdb(hea_path_1 + str(i) + ".hea")
    signal_orig = record_1.get_lead("MLII")
    if signal_orig == None:
        print(i)
        continue
    signal_original.append(signal_orig)

length = len(signal_original)
print(length)


#######################################################################################################
#                                     ADDING THE NOISE AND SAVING THE SIGNAL 
#######################################################################################################

i = 0, k = 0
with open('x_train.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    for k in range(length):
        print(k)
        while i<len(signal_original[k]) - 1024:
            j = random.randint(0,len(signal_noise) - 1025)
            noise_temp_signal = [a + b for a, b in zip(signal_original[k][i:i+1024], signal_noise[j:j+1024])]
            print(len(noise_temp_signal))
            writer.writerow(noise_temp_signal)
            print(i,k)
            i = i+ 1024
        i = 0

i = 0, k = 0
with open("y_train.csv", "w", newline = '') as file:
    writer = csv.writer(file)
    for k in range(length):
        print(k)
        while i<len(signal_original[k])-1024:
            original_temp_signal = signal_original[k][i:i+1024]
            print(len(original_temp_signal))
            writer.writerow(original_temp_signal)
            print(i,k)
            i = i+ 1024
        i = 0


##########################################################################################################
#                                         PLOTTING THE SIGNALS
##########################################################################################################
# Plotting the signal
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Heart_Sound_Data')
ax1.plot(signal_original[0:1024])
ax2.plot(signal_noise[0:1024])
plt.show()





import torch
