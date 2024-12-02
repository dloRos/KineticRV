c = 3e8;             %light velocity in free space
%------参数1-------%
fc1_start = 77e9;           %center frequency
lambda1 = c/fc1_start;       % wave length
ti1 = 5e-6;     %idle_time
ts1 = 44.7e-6;    %sweep time 
tm1 = ti1+ts1;  %PRI
global sweep_slope1;
sweep_slope1 = 46e12;  % slope 62.5Mhz/us
B1 = sweep_slope1*ts1;
fs1 = 6250000;                  %sampling frequency 6.25MHz
ADC_sample_num1 = 256;
rang_max1 = (fs1/sweep_slope1)*c/2;
range_resolution = rang_max1/ADC_sample_num1;

DeviceID = 1642;
numChirpPerLoop1 = 1;
Start_frameIdx = 1;
End_frameIdx =15625;
numLoops=End_frameIdx-Start_frameIdx+1;
numRXPerDevice = 4;

tframe = 0.16*10e-4;
t = (0:End_frameIdx-1)*tframe;

fdmax = 1/2/tframe;
vmax = fdmax * lambda1 /2;

[radar_data_temp1] = Multi_readAdcData(fileFullPath, DeviceID, Start_frameIdx, End_frameIdx,ADC_sample_num1, numChirpPerLoop1, numLoops, numRXPerDevice);
radar_data_Rx1 = radar_data_temp1(:,1:14000,1);
radar_data_Rx = sum(radar_data_temp1, 3)