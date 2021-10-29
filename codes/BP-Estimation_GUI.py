# In[] import 函式庫
from tkinter import Tk, Label, StringVar, Button, Entry
from tkinter.filedialog import askopenfilename
from numpy import linspace
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image,ImageTk
import numpy as np
from statistics import mean
import scipy.io, math
from keras.models import load_model
import time

# In[] Functions
# 資料處理
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


def getTrainingInput(segments, answers, M):
    M_train_X, M_train_y = list(), list()
    
    for i in range(len(segments)-M+1):
        M_train_input = [x for x in segments[i:i+M]]
        M_train_X.append(M_train_input)
        
        M_train_output = [x for x in answers[i:i+M]]
        M_train_y.append(M_train_output)
    
    return M_train_X, M_train_y

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


# 介面功能
def browse_btn_hit():
    global var_filePath
    filePath = askopenfilename()
    var_filePath.set(filePath)


def start_btn_hit():
    global var_filePath, sub_entry, model, window, pre_data, filePath, stopFlag
    stopFlag = False
    sbp_pred_var.set('')
    sbp_gt_var.set('')
    dbp_pred_var.set('')
    dbp_gt_var.set('')
 
    
    subject_n = int(sub_entry.get()) - 1
    
    n_timesteps = 10
    
    X, y, peakIndexs = extractFeature(pre_data, subject_n, n_timesteps)
    
    
    X = np.concatenate([X], axis=0)
    y = np.concatenate([y], axis=0)
    yTen = np.array([p[n_timesteps-1] for p in y])
    
    pred = model.predict(X)
    predTen = np.array([p[n_timesteps-1] for p in pred])
    
    
    w = linspace(0, 6, num=750)
    
    peakIndexs = [p + 750 for p in peakIndexs]
    startPeak = 11
    startPeakIndex = peakIndexs[startPeak]
    predCount = 0
    
    
    
    ppg_signal = np.concatenate([linspace(0, 0, num=750), pre_data[subject_n][0], linspace(0, 0, num=800)], axis=0)
    
    ecg_signal = np.concatenate([linspace(0, 0, num=750), pre_data[subject_n][1], linspace(0, 0, num=800)], axis=0)
    
    
    
    moveLen = 15
    speed = moveLen/1000
    
    shift = int(len(pre_data[subject_n][0]) / moveLen) + 3 + 50
    
    peakIndexs.append(peakIndexs[-1] + moveLen)
    for i in range(shift):
        if stopFlag:
            break
        
        s = i * moveLen
        e = i * moveLen + len(w)
        
        Line = (s + int(len(w)/2))
        if (Line >= startPeakIndex) and (startPeak < len(peakIndexs)-1):
            startPeak += 1
            startPeakIndex = peakIndexs[startPeak]
            sbp_pred_var.set(round(predTen[predCount][0], 3))
            sbp_gt_var.set(round(yTen[predCount][0], 3))
            dbp_pred_var.set(round(predTen[predCount][1], 3))
            dbp_gt_var.set(round(yTen[predCount][1], 3))
            predCount += 1
        
        
        for j, index in enumerate(peakIndexs):
            if index > s:
                peakStart = j
                break
            
        for j, index in enumerate(peakIndexs):
            if (index >= e):
                peakEnd = j
                break
            
            
        newPeaks = [p - s for p in peakIndexs[peakStart:peakEnd]]
        
        plotPPG(w, ppg_signal[s:e], newPeaks)
        plotECG(w, ecg_signal[s:e])
        window.update()
        time.sleep(speed)
    


def plotPPG(x, y, peaks):
    global window

    # 畫圖
    f = Figure(figsize=(7.5,3.5), dpi=100)
    a = f.add_subplot(111)
    # 绘制图形
    a.plot(x, y, c='C0')
    
    peakLoc = [x[i] for i in peaks]
    peakAmp = [y[i] for i in peaks]
    a.plot(peakLoc, peakAmp, 'g^', color='C1')
    
    a.plot([3, 3], [0, 1], c='C1')
    
    a.set_xlim([0, 6])
    a.set_ylim([0, 1])
    a.set_title('PPG Signal')
    a.set_xlabel('Seconds')
    a.set_ylabel('Amplitude')
    canvas = FigureCanvasTkAgg(f, master=window)
    canvas.get_tk_widget().place(x=750, y=410, anchor='nw')


def plotECG(x, y):
    global window

    # 畫圖
    f = Figure(figsize=(7.5,3.5), dpi=100)
    a = f.add_subplot(111)
    # 绘制图形
    a.plot(x, y, c='C0')
    a.plot([3, 3], [0, 1], c='C1')
    
    a.set_xlim([0, 6])
    a.set_ylim([0, 1])
    a.set_title('ECG Signal')
    a.set_xlabel('Seconds')
    a.set_ylabel('Amplitude')
    canvas = FigureCanvasTkAgg(f, master=window)
    canvas.get_tk_widget().place(x=750, y=20, anchor='nw')
    
    
def stop_btn_hit():
    global stopFlag
    stopFlag = True


# In[] 使用介面設計
if __name__ == '__main__':
    # 全域變數 
	
	# 檔案路徑
    path1 = r'..\Preprocessed_Part_1.mat'
    path3 = r'..\Preprocessed_Part_3.mat'
    
    print('Loading data...')
    pre_data = np.concatenate([scipy.io.loadmat(path1)['preprocessed_Part_1'][0], scipy.io.loadmat(path3)['preprocessed_Part_3'][0]], axis=0)
    print('Loadin Finish !!')
    
	
	print('Loading model...')
	# Model Path
    modelPath = r'..\ann_lstm_lstm.h5'
    model = load_model(modelPath, compile=False)
    
    filePath = ''
    stopFlag = False
    
    
    # 建立視窗window
    window = Tk()
    
    # 給視窗的視覺化起名字
    window.title('BP_Estimation')
    
    # 設定視窗的大小(長 * 寬)
    window.geometry('1600x800')  # 這裡的乘是小x
    
    imgPath = r'E:\NCU_CS\研究所修課\機器學習\Project\data\hert_beat.jpg'
    img = Image.open(imgPath)
    imgTk = ImageTk.PhotoImage(img)
    imgLabel = Label(window, image=imgTk)
    imgLabel.place(x=100, y=430)
    
    
    

    
    # Subject 的標示
    sub_label = Label(window, text='Subject:', font=('Arial', 20), width=10, height=2)
    sub_label.place(x=10, y=50, anchor='nw')
    # Subject 的輸入
    sub_entry = Entry(window, width=6, font=('Arial', 20))
    sub_entry.place(x=148, y=73, anchor='nw')
    
    

    
    # "開始"按鍵
    start_btn = Button(window, text='BP Estimation', font=('Arial', 20), width=12, height=1, command=start_btn_hit)
    start_btn.place(x=250, y=60, anchor='nw')

    
    # "停止"按鍵
    stop_btn = Button(window, text='Stop', font=('Arial', 20), width=12, height=1, command=stop_btn_hit)
    stop_btn.place(x=470, y=60, anchor='nw')
    
    
    
    # "SBP"的標示
    sbp_label = Label(window, text='SBP Estimation', font=('Arial', 20), width=15, height=1)
    sbp_label.place(x=0, y=235, anchor='nw')
    
    
    
    # "DBP"的標示
    dbp_label = Label(window, text='DBP Estimation', font=('Arial', 20), width=15, height=1)
    dbp_label.place(x=0, y=320, anchor='nw')
    
    
    # 預測的標示
    pred_label = Label(window, text='Predicted', font=('Arial', 20), width=15, height=1)
    pred_label.place(x=220, y=160, anchor='nw')
    
    
    
    # 真實的標示
    gt_label = Label(window, text='GT', font=('Arial', 20), width=3, height=1)
    gt_label.place(x=540, y=160, anchor='nw')
    
    
    # "SBP 預測" 變數顯示的地方
    sbp_pred_var = StringVar()
    sbp_pred_label = Label(window, textvariable=sbp_pred_var, bg='white', fg='black',font=('Arial', 20), width=11, height=2)
    sbp_pred_label.place(x=260, y=220, anchor='nw')
    
    
    # "SBP 真實" 變數顯示的地方
    sbp_gt_var = StringVar()
    sbp_gt_label = Label(window, textvariable=sbp_gt_var, bg='white', fg='black',font=('Arial', 20), width=11, height=2)
    sbp_gt_label.place(x=475, y=220, anchor='nw')
    
    
    
    # "DBP 預測" 變數顯示的地方
    dbp_pred_var = StringVar()
    dbp_pred_label = Label(window, textvariable=dbp_pred_var, bg='white', fg='black',font=('Arial', 20), width=11, height=2)
    dbp_pred_label.place(x=260, y=310, anchor='nw')
    

    
    # "DBP 真實" 變數顯示的地方
    dbp_gt_var = StringVar()
    dbp_gt_label = Label(window, textvariable=dbp_gt_var, bg='white', fg='black',font=('Arial', 20), width=11, height=2)
    dbp_gt_label.place(x=475, y=310, anchor='nw')
    

    plotPPG([], [], [])
    plotECG([], [])
    
    # 主視窗迴圈顯示
    window.mainloop()


