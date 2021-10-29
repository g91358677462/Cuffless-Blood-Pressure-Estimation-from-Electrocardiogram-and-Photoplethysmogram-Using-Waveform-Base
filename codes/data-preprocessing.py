# In[ ]:
import numpy as np
import scipy.io, math
from statistics import mean


# In[ ]:
# Data Path
path1 = r'..\Preprocessed_Part_1.mat'
path3 = r'..\Preprocessed_Part_3.mat'

print('Loading Data...')
pre_data = scipy.io.loadmat(path1)['preprocessed_Part_1'][0]
pre_data2 = scipy.io.loadmat(path1)['preprocessed_Part_3'][0]

print('Loading Finish !!')

# In[ ]:
# find the maximum in each window
def findPeaks1st(data, window_size):
    peaks = []
    for i in range(0, len(data)//window_size + 1):
        segment_data = data[i*window_size: (i+1)*window_size]
        if len(segment_data) == 0: break
        
        max_value = np.amax(segment_data)
        segment_data = [0 if x != max_value else max_value for x in segment_data]
        peaks += segment_data
    return peaks

# remove small values using a threshold filter
def findPeaks2nd(peaks):
    values = list(filter(lambda x: x > 0, peaks))
    threshold = mean(values) - 0.25
    
    for i, value in enumerate(peaks):
        if value < 0: continue
        if value > threshold: continue
        peaks[i] = 0
    return peaks

def findPeaks3rd(data):
    diff_window_results = [findPeaks2nd(findPeaks1st(data, window_size))
                           for window_size in range(50, 66)]
    peaks = list(map(sum, zip(*diff_window_results)))
    peaks = [x/15 for x in peaks]
    return peaks

def findPeaks4th(peaks):
    threshold = 0.24
    peaks = [0 if x < threshold else 1 for x in peaks]
    return peaks


def overSampling2(data):
    r1 = len(data)/256
    r2 = 1 - r1
    new_data = []
    
    for i in range(256):
        sampling_index = math.floor(i*r1)
        new_data.append(data[sampling_index])
    
    for i in range(256-2):
        if i % 2 == 1: continue
            
        new_data[i+1] = new_data[i]*r2 + new_data[i+2]*r1
    return new_data



# In[ ]:
def getTrainingInput(segments, answers, M):
    M_train_X, M_train_y = list(), list()
    
    for i in range(len(segments)-M):
        M_train_input = [x for x in segments[i:i+M]]
        M_train_X.append(M_train_input)
        
        M_train_output = [x for x in answers[i:i+M]]
        M_train_y.append(M_train_output)
    
    return M_train_X, M_train_y


# In[ ]:
'''
0	1st row	: preprocessed PPG
1	2nd row	: preprocessed ECG
2	3rd row	: ABP
'''
def extractFeature(pre_data, subject_n, n_timesteps):
    ppg_signal = pre_data[subject_n][0]
    ecg_signal = pre_data[subject_n][1]
    abp_signal = pre_data[subject_n][2]

    ppg_peaks = findPeaks4th(findPeaks3rd(ppg_signal))
    peak_index = [i for i, j in enumerate(ppg_peaks) if j == 1]

    segments, answers = list(), list()
    for i in range(len(peak_index)-2):
        start, end = peak_index[i], peak_index[i+2] + 1

        new_segment = overSampling2(ecg_signal[start:end]) + overSampling2(ppg_signal[start:end]) + [(end-start)/256]
        segments.append(new_segment)

        sbp = max(abp_signal[start:end])
        dbp = min(abp_signal[start:end])
        answers.append([sbp, dbp])  
    M_train_X, M_train_y = getTrainingInput(segments, answers, M=n_timesteps)
    
    
    return M_train_X, M_train_y, peak_index
    



n_timesteps = 10

# In[ ]:
# 70% training data (part1 0~3000)
train_X1 = np.concatenate([extractFeature(pre_data, subject_n, n_timesteps, get='X') 
                            for subject_n in range(0, 3000)], axis=0)
train_y1 = np.concatenate([extractFeature(pre_data, subject_n, n_timesteps, get='y')
                           for subject_n in range(0, 3000)], axis=0)


# In[ ]:

# 70% training data (part2 0~1200)
train_X2 = np.concatenate([extractFeature(pre_data2, subject_n, n_timesteps, get='X')
                           for subject_n in range(0, 1200)], axis=0)
train_y2 = np.concatenate([extractFeature(pre_data2, subject_n, n_timesteps, get='y')
                           for subject_n in range(0, 1200)], axis=0)


# In[ ]:


train_X = np.concatenate([train_X1, train_X2], axis=0)
np.save('M10_train_X.npy', train_X)

train_y = np.concatenate([train_y1, train_y2], axis=0)
np.save('M10_train_y.npy', train_y)



# In[ ]:
# 10% validation data (part2 1200~1800)
val_X = np.concatenate([extractFeature(pre_data2, subject_n, n_timesteps, get='X')
                        for subject_n in range(1200, 1800)], axis=0)
val_y = np.concatenate([extractFeature(pre_data2, subject_n, n_timesteps, get='y')
                        for subject_n in range(1200, 1800)], axis=0)


# In[ ]:
np.save('M10_val_X.npy', val_X)
np.save('M10_val_y.npy', val_y)



# In[ ]:
# 20% testing data (part2 1800~3000)
test_X = np.concatenate([extractFeature(pre_data2, subject_n, n_timesteps, get='X')
                         for subject_n in range(1800, 3000)], axis=0)
test_y = np.concatenate([extractFeature(pre_data2, subject_n, n_timesteps, get='y')
                         for subject_n in range(1800, 3000)], axis=0)


# In[ ]:
np.save('M10_test_X.npy', test_X)
np.save('M10_test_y.npy', test_y)



# In[ ]:
n_timesteps = 10
all_train_X, all_train_y = list(), list()

# 70% training data
for subject_n in range(0, 3000):
    ppg_signal = pre_data[subject_n][0]
    ecg_signal = pre_data[subject_n][1]
    abp_signal = pre_data[subject_n][2]

    ppg_peaks = findPeaks4th(findPeaks3rd(ppg_signal))
    peak_index = [i for i, j in enumerate(ppg_peaks) if j == 1]

    segments, answers = list(), list()
    for i in range(len(peak_index)-2):
        start, end = peak_index[i], peak_index[i+2] + 1

        new_segment = overSampling2(ecg_signal[start:end]) + overSampling2(ppg_signal[start:end]) + [(end-start)/256]
        segments.append(new_segment)

        sbp = max(abp_signal[start:end])
        dbp = min(abp_signal[start:end])
        answers.append([sbp, dbp])
    
    if subject_n % 100 == 0: print(subject_n)
    
    if subject_n == 0:
        all_train_X, all_train_y = getTrainingInput(segments, answers, M=n_timesteps)
    
    M_train_X, M_train_y = getTrainingInput(segments, answers, M=n_timesteps)
    
    if len(M_train_X) == 0 or len(M_train_y) == 0: continue
    
    all_train_X += M_train_X
    all_train_y += M_train_y


# In[ ]:
np.save('M10_train_X.npy', np.array(all_train_X, dtype='float16'))
np.save('M10_train_y.npy', np.array(all_train_y, dtype='float16'))


# In[ ]:
# 70% training data
all_train_X2, all_train_y2 = list(), list()
for subject_n in range(0, 1200):
    ppg_signal = pre_data2[subject_n][0]
    ecg_signal = pre_data2[subject_n][1]
    abp_signal = pre_data2[subject_n][2]

    ppg_peaks = findPeaks4th(findPeaks3rd(ppg_signal))
    peak_index = [i for i, j in enumerate(ppg_peaks) if j == 1]

    segments, answers = list(), list()
    for i in range(len(peak_index)-2):
        start, end = peak_index[i], peak_index[i+2] + 1

        new_segment = overSampling2(ecg_signal[start:end]) + overSampling2(ppg_signal[start:end]) + [(end-start)/256]
        segments.append(new_segment)

        sbp = max(abp_signal[start:end])
        dbp = min(abp_signal[start:end])
        answers.append([sbp, dbp])
    
    if subject_n % 100 == 0: print(subject_n)
    
    M_train_X, M_train_y = getTrainingInput(segments, answers, M=n_timesteps)
    
    if len(M_train_X) == 0 or len(M_train_y) == 0: continue
    
    all_train_X2 += M_train_X
    all_train_y2 += M_train_y


# In[ ]:
np.save('M10_train_X2.npy', np.array(all_train_X2, dtype='float16'))
np.save('M10_train_y2.npy', np.array(all_train_y2, dtype='float16'))


# In[ ]:
all_val_X, all_val_y = list(), list()

# 10% validation data
for subject_n in range(1200, 1800):
    ppg_signal = pre_data2[subject_n][0]
    ecg_signal = pre_data2[subject_n][1]
    abp_signal = pre_data2[subject_n][2]

    ppg_peaks = findPeaks4th(findPeaks3rd(ppg_signal))
    peak_index = [i for i, j in enumerate(ppg_peaks) if j == 1]

    segments, answers = list(), list()
    for i in range(len(peak_index)-2):
        start, end = peak_index[i], peak_index[i+2] + 1

        new_segment = overSampling2(ecg_signal[start:end]) + overSampling2(ppg_signal[start:end]) + [(end-start)/256]
        segments.append(new_segment)

        sbp = max(abp_signal[start:end])
        dbp = min(abp_signal[start:end])
        answers.append([sbp, dbp])
    
    if subject_n % 100 == 0: print(subject_n)
    
    if subject_n == 0:
        all_val_X, all_val_y = getTrainingInput(segments, answers, M=n_timesteps)
        
    M_val_X, M_val_y = getTrainingInput(segments, answers, M=n_timesteps)
    
    if len(M_val_X) == 0 or len(M_val_y) == 0: continue
    
    all_val_X += M_val_X
    all_val_y += M_val_y


# In[ ]:
np.save('M10_val_X.npy', np.array(all_val_X, dtype='float16'))
np.save('M10_val_y.npy', np.array(all_val_y, dtype='float16'))


# In[ ]:
all_test_X, all_test_y = list(), list()

# 20% testing data
for subject_n in range(1800, 3000):
    ppg_signal = pre_data2[subject_n][0]
    ecg_signal = pre_data2[subject_n][1]
    abp_signal = pre_data2[subject_n][2]

    ppg_peaks = findPeaks4th(findPeaks3rd(ppg_signal))
    peak_index = [i for i, j in enumerate(ppg_peaks) if j == 1]

    segments, answers = list(), list()
    for i in range(len(peak_index)-2):
        start, end = peak_index[i], peak_index[i+2] + 1

        new_segment = overSampling2(ecg_signal[start:end]) + overSampling2(ppg_signal[start:end]) + [(end-start)/256]
        segments.append(new_segment)

        sbp = max(abp_signal[start:end])
        dbp = min(abp_signal[start:end])
        answers.append([sbp, dbp])
        
    if subject_n % 100 == 0: print(subject_n)
    
    if subject_n == 0:
        all_test_X, all_test_y = getTrainingInput(segments, answers, M=n_timesteps)
        
    M_test_X, M_test_y = getTrainingInput(segments, answers, M=n_timesteps)
    
    if len(M_test_X) == 0 or len(M_test_y) == 0: continue
    
    all_test_X += M_test_X
    all_test_y += M_test_y


# In[ ]:
np.save('M10_test_X.npy', np.array(all_test_X, dtype='float16'))
np.save('M10_test_y.npy', np.array(all_test_y, dtype='float16'))

