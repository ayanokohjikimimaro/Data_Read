import datetime
import os
import sys
import pandas as pd
import numpy as np

"""
# =============================================================================
# file name
# =============================================================================
now = datetime.datetime.now()

label = input("Please input the Label: ")
#label = 'footsteps'

print('filename : acc' + '_' + label + '_{0:%y%m%d_%H%M%S}'.format(now)) 
filename = 'acc' + '_' + label + '_{0:%y%m%d_%H%M%S}'.format(now)

# =============================================================================
# *.txt gen
# =============================================================================
dirpath = os.path.dirname(os.path.abspath(sys.argv[0]))
filepath = dirpath + "\\" + "environmental_sound"
f = open(filepath +"\\"+ filename +".txt",'w')
f.close()

# =============================================================================
# Data paste
# =============================================================================
while True:
    label = input("Paste the Data in this file ??: ")
    if label == "OK":
        print ("Perform the next processing")
        break
    elif label == "C":
        sys.exit()
    else:
        print("Try Again ")
"""

# =============================================================================
# Data Scan    
# =============================================================================
filepath = "C:\\Users\\Satoshi KIMURA\\Documents\\python\\environmental_sound\\"
filename = "acc_foot_181014_220621"
dataset = pd.read_csv(filepath +"\\"+ filename +".txt", header=None)

seach_word = '----------------'
end_search = [i for i, x in enumerate(dataset[0]) if x == seach_word]
data_top_flg = end_search[-2]
data_end_flg = end_search[-1]

dataset_ex = dataset[(data_top_flg+1):(data_end_flg-1)]

# =============================================================================
# Data Conversion
# =============================================================================
time_int = np.array([int(n) for n in dataset_ex[0]])
time_ini = time_int - time_int[0]

data_int = np.array([int(n) for n in dataset_ex[1]])

for iii in range(len(data_int)):
    if data_int[iii] > 2**15-1:
        data_int[iii] = data_int[iii] - 2**16
    else:
        pass
    