import os
import numpy as np


# Y = np.load('Y_test_Driving_NewApproach_Injected_v2.npz')
Y = np.load('dataY.npz')

Y= Y.f.arr_0

# X = np.load('X_test_Driving_NewApproach_Injected_v2.npz')
X = np.load('dataX.npz')

X = X.f.arr_0

import gc
gc.collect()

# import Random Forest classifier
import time

from sklearn.ensemble import RandomForestClassifier
import pickle
file_name = "rf.pkl"
rfc = pickle.load(open(file_name, "rb"))


start_t = time.time()
y_pred = rfc.predict(X)
end = time.time()

print("Runtime of the program is ")
print({end - start_t})
total_time = end - start_t
timesample = total_time/Y.shape[0]
aux_inference = timesample*1000000
print("    us/sample is ")
print({aux_inference})
