import pickle
import numpy as np
import csv

'''
# Load the pickle file
with open('.\\numerical_results\\MyBinance\\SigWGAN_LogSigRNN_0\\x_real_test.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.shape)
data = data.tolist()
print(len(data), len(data[0]), len(data[0][0]))

# to write file you saved
with open('x_real.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(data)
'''

# to read file you saved
with open('x_real.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader) # length: Number of windows

windows = []
for window in data:
    timings = []
    for timing in window:
        timings.append(eval(timing))  # length of [eval(timing)]: number of assets
    windows.append(timings)

result = np.array(windows)
 
print(result.shape)