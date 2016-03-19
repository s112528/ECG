close all;
clear all;

path_begin = 'C:\Users\Florent\Documents\ULG\2ème Master\Master thesis\Data\PSGData\';
patient_nb = '8';
patient_session = '1';
patient_name = strcat('Sujet',patient_nb,'-',patient_session);
path = strcat(path_begin,'Sujet', patient_nb,'\', patient_session,'\chk1_Traces');
% % Example of data of the ULG
%D=spm_eeg_load('chk1_Traces.mat'); 
D=spm_eeg_load(strcat(path,'.mat')); 
%display(D) % To give some info
%chanlabels(D) % To know which channel is what
t1 = 0; % time in [s]
t2 = 60*10; % time in [s]
ECG=D.selectdata('ECG',[t1 t2],[]); % para : channel, [t1 (s) t2(s)],[] = ?
fs = 512;
my_gain = 1.4;
delta_n = 15;
time = (t1:1/fs:t2);

% http://dsp.stackexchange.com/questions/16394/removing-baseline-drift-from-ecg-signal
% b = [1 -1];
% a = [1 -0.99];
% ECG0 = filter(b,a,ECG0);

% 50Hz Notch filter
% f_noise = 50; % frequency of the interference
% Q = 50; % quality factor
% Wo = f_noise/(fs/2);
% BW = Wo/Q;
% [b,a] = iirnotch(Wo,BW);
% ECG = filter(b,a,ECG0);

% f = fdesign.notch('N,F0,Q',26,50,10,fs);
% Hd = design(f);
% ECG = filter(Hd,ECG0);
% %ECG = filter(Hd,ECG);
% %ECG = filter(Hd,ECG);
% figure;
% subplot(2,1,1);
% plot(time,ECG0);
% title('No notch filter');
% subplot(2,1,2);
% plot(time,ECG);
% title('Notch filter');

[real_peaks,~] = fctReadData(ECG, t1, t2, fs, my_gain, delta_n);

dt_int = 60;

[mean_RR, SDNN, mean_HR, mean_HRI, RMSSD, NN50, pNN50, SD1, SD2,...
    LF_power, HF_power, LFoverHF, LF_norm, HF_norm, LF_f_max,HF_f_max, dt_tot, Pxx, pxx1]= ...
    fctComputeFeatures( t1, t2, fs, dt_int, real_peaks);


%names = {'mean_RR', 'SDNN', 'mean_HR', 'RMSSD', 'NN50', 'pNN50', ...
%    'LF_power', 'HF_power', 'LFoverHF', 'LF_norm', 'HF_norm', ...
%    'LF_f_max', 'HF_f_max'};

names = {'mean_RR', 'SDNN', 'mean_HR', 'mean_HRI', 'RMSSD', 'NN50', ...
    'pNN50', 'SD1', 'SD2', 'LF_power', 'HF_power', 'LFoverHF', 'LF_norm', 'HF_norm', ...
    'LF_f_max', 'HF_f_max'};

features_table = num2cell([mean_RR; SDNN; mean_HR; mean_HRI; RMSSD; ...
    NN50; pNN50; SD1; SD2; LF_power; HF_power; LFoverHF; LF_norm; HF_norm; ...
    LF_f_max; HF_f_max].');

filename = strcat(pwd,'\ExcelFilesFeatures\',patient_name,'.xls');
dataExcel=[ names; features_table];  % To have the name of each column at the top
% To NOT have the name of each column at the top, simply delete num2cell
% and just do dataExcel = features_table;
xlswrite(filename,dataExcel );

