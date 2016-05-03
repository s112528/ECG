%Fct that reads the ECG data & returns the peaks amplitudes and locations

function [real_peaks, ECG_FIR] = fctReadData(ECG, t1, t2, fs, my_gain, delta_n)
    if t2 < t1
        error('Error : Invalid argument time t2 must be after time t1');
    end
    time = (t1:1/fs:t2); 
    N = length(time);
    
    %f = fdesign.notch('N,F0,Q',26,50,10,fs);
	%Hd = design(f);
    %ECG_notch = filter(Hd,ECG);
    %hold on;
    %plot(time, ECG_notch,'g');
    
    % FIR filter
    %order = 40;
    % Normalized frequency : https://www.mathworks.com/matlabcentral/newsreader/view_thread/286192
    %w_inf = (2*6)/fs; % 8Hz in normalized frequency
    %w_sup = (2*18)/fs; % 20Hz in normalized frequency
    %b = fir1(order,[w_inf w_sup], kaiser(order+1,0.5)); % FIR filter between 8 and 20Hz, fs = 512 Hz, w = 2*pi*f/fs
    %b = fir1(order,[w_inf w_sup],'bandpass');
    %b = fir1(order,[w_inf w_sup],'bandpass');
    %[b,a] = cheby1(4,5,[w_inf w_sup],'bandpass');
    %ECG_FIR = filter(b,1,ECG);
   
    %fcuts = [5 6 18 19];
    fcuts = [7 8 20 21]; % seems to be ok too
    mags = [0 1 0];
    %devs = [0.01 0.05 0.01];
    devs = [0.0315 0.1 0.0315]; % 0.0315 => 30dB of attenuation between stopband and passband and riple of 10% in pass band

    [n,Wn,beta,ftype] = kaiserord(fcuts,mags,devs,fs);
    fprintf('The order of the filter is equal to %d \n', n);
    b = fir1(n,Wn,ftype,kaiser(n+1,beta),'noscale');
    
    [H,f] = freqz(b,1,1024,fs);
    figure;
    plot(f,abs(H));
    xlim([0 30]);
    xlabel('Frequency (Hz)');
    ylabel('|H|');
    title('Impulse response of the filter');
    grid;
    set(gca,'FontSize', 22);
    
    %ECG_FIR = filtfilt(b,1,ECG); % filtfilt => |H|^2 => 2*attenuation of fir1 and no phase lag
    ECG_FIR = filter(b,1,ECG);
    grpdelay(b,N,fs)
    delay = mean(grpdelay(b)); % The delay is in samples 
    fprintf('Loss of %d s of the signal due to the filter \n', delay/fs);
    
    % To compensate the delay
    ECG = ECG(1:end-(delay));
    time = time(1:end-(delay));
    ECG_FIR(1:delay) = [];
    N = length(time);
    t2 = t2-(delay/fs);
    
%     figure;
%     plot(time, ECG,'b', time, ECG_FIR,'g');
%     xlabel('time [s]');
%     ylabel('ECG [mv]');
%     %title('Impact of the FIR filter');
%     legend('Raw ECG', 'Filtered ECG');
%     grid;
%     set(gca,'FontSize', 22);
    
    figure;
    plot(time, ECG,'b', time, ECG_FIR,'g','linewidth',2.0);
    xlabel('Time [s]');
    ylabel('ECG [mV]');
    %title('Impact of the FIR filter');
    legend('Raw ECG', 'Filtered ECG');
    grid;
    set(gca,'fontsize',22,'fontname','Times', 'LineWidth',0.5);
    set(gca,'XLim',[235 255]);
    set(gcf, 'paperunits', 'inches');
    Lx=8; Ly=6;
    set(gcf, 'papersize', [Lx Ly]);
    set(gcf, 'PaperPosition', [0.01*Lx 0.01*Ly 1.05*Lx 1.02*Ly]);
    name = strcat('C:\Users\Florent\Documents\ULG\2ème Master\Master thesis\Rapport\Images\MotionArtifactsFiltered');
    %print(name,'-dpdf')
    
    figure;
    subplot(2,2,1);
    plot(time, ECG);
    xlabel('time (s)');
    ylabel('ECG');
    title('Signal brut'); 
    hold on;
    plot(time, ECG_FIR,'g');
    subplot(2,2,2);
    plot(time, ECG_FIR);
    xlabel('time (s)');
    ylabel('ECG');
    title('FIR filter');
    
    % 1st derivative
    
    dECG = zeros(1,length(ECG));
    dECG(2:(length(ECG)-1)) = (fs/2)*( ECG_FIR((2:(length(ECG)-1))+1) -  ECG_FIR((2:(length(ECG)-1))-1)); % Avoid having for i = 2:(length(ECG)-1)
    dECG(1) = fs*(ECG_FIR(2) - ECG_FIR(1));
    dECG(end) = fs*(ECG_FIR(end) - ECG_FIR((end-1)));
    subplot(2,2,3);
    plot(time, dECG);
    xlabel('time (s)');
    ylabel('dECG');
    title('1st derivative');
    
    
    % Hilbert Direct
    H = hilbert(dECG);
    subplot(2,2,4);
    plot(time,abs(H));
    title('Hilbert')
    xlabel('time (s)');
    ylabel('H(ECG)');
    
    
    % Bode diagram of the filter 
    figure;
    freqz(b,1);
    %H1 = tf(b,1);
    %bodeplot(H1);
    title('Impulse response of the FIR filter');
    
    % First stage detector
    
    % 1) Find the peaks
    peaks = []; % vector nx3 where row 1 = peak, row 3 = its location in time [s] and row 3 = its location in sample term [-]
    index_start = 1;
    index_stop = round((fs/360)*1024);
    %index_stop = 1024;
    inc = round((fs/360)*1024);
    threshold = zeros(size(ECG));
    prev_max = 10^9;
    nb_inc = 0;
    while index_stop < N
        sub_H = abs(H(index_start:index_stop));
        sub_H_rms = rms(sub_H);
        sub_H_max = max(sub_H);
        
        % find the threshold
        if sub_H_rms >= (0.18*sub_H_max) % Level of noise high
            if sub_H_max > 2*prev_max
                threshold(index_start:index_stop) = (0.39*prev_max)*my_gain;
            else
                threshold(index_start:index_stop) = (0.39*sub_H_max)*my_gain;
            end
        else % Low level of noise
            threshold(index_start:index_stop) = (1.6*sub_H_rms)*my_gain;
        end
        % find the peaks
        [pks,locs] = findpeaks(sub_H,'MINPEAKHEIGHT',threshold(index_stop));
        if sub_H(end) > threshold(index_stop)
            pks = [pks sub_H(end)];
            locs = [locs inc];
        end
        locs = locs + (nb_inc*inc);
        peaks = [peaks [pks;locs]];
        
        prev_max = sub_H_max;
        index_start = index_start + inc;
        index_stop = index_stop + inc;
        nb_inc = nb_inc + 1;
    end
    
    sub_H = zeros(1, (length(H(index_start:N)) + 1));
    sub_H(1:end-1) = abs(H(index_start:N));
    sub_H(end) = sub_H(end-1)-1;
    sub_H_rms = rms(sub_H);
    sub_H_max = max(sub_H);
    
    if sub_H_rms >= (0.18*sub_H_max) % Level of noise high
        if sub_H_max > 2*prev_max
            threshold(index_start:N) = 0.39*prev_max*my_gain;
        else
            threshold(index_start:N) = 0.39*sub_H_max*my_gain;
        end
    else % Low level of noise
        threshold(index_start:N) = 1.6*sub_H_rms*my_gain;
    end
    
    % find the peaks
    [pks,locs] = findpeaks(sub_H,'MINPEAKHEIGHT',threshold(N));
    if ~isempty(locs)
        locs = locs + (nb_inc*inc);
        if locs(end) ~= N % if the last peak is ~= from the last value
            peaks = [peaks [pks;locs]]; % peaks found are OK
        else % if not it means that last point found as max because fct is varying until extremum
            if length(locs) ~= 1 
                peaks = [peaks [pks(1:end-1);locs(1:end-1)]]; % We take all the max until end-1 which are good max. Last one is not a good one
            end
        end
    end
  
    peaks_time = t1 + (peaks(2,:)-1)/fs; % We put peaks locations in [s]
    peaks = [peaks(1,:); peaks_time; peaks(2,:)]; % peaks = nx3
    % figure;
    % plot(time,abs(H),'b',time,threshold,'k');
    % % http://nl.mathworks.com/help/matlab/ref/colorspec.html
    % hold on;
    % plot(peaks(2,:),peaks(1,:),'.r');
    % legend('Hilbert', 'Threshold', 'Peaks');
    % xlabel('time(s)');
    %
    figure;
    plot(time,ECG,'b');
    % http://nl.mathworks.com/help/matlab/ref/colorspec.html
    hold on;
    plot(peaks_time(:),ECG(peaks(3,:)),'.r');
    title('First stage peak detector');
    legend('ECG','Peaks');
    xlabel('time(s)');
    
    % 2) Delete the too closed peaks
    last_peaks = peaks(:,1);
    nb_true_peaks = 0;
    mean_dt_peaks = 0;
    mean_ampl_peaks = 0;
    true_peaks = [];
    
    if length(peaks(1,:))>=2 % If there is at least 2 peaks in the ECG
        for i = 2:length(peaks(1,:))
            if((peaks(2,i) - last_peaks(2)) <= 0.4) %  If dt < 400ms
                if (nb_true_peaks < 3) % If less than 3 true peaks
                    if (peaks(1,i) > last_peaks(1)) % We keep the highest one
                        last_peaks = peaks(:,i);
                    end
                else % If more than 3 true peaks
                    % if new pics closer in ampl_mean & dt_mean to the last
                    % true peak than the last peak
                    if (abs(peaks(1,i)-mean_ampl_peaks) + abs(peaks(2,i)-mean_dt_peaks)<(abs(last_peaks(1)-mean_ampl_peaks) + abs(last_peaks(2)-mean_dt_peaks))) %  % if new peak closer in ampl_mean & dt_mean to the last true peak than the last peak
                        last_peaks = peaks(:,i);
                    %elseif abs(peaks(1,i)-mean_ampl_peaks) < (abs(last_peaks(1)-mean_ampl_peaks)) % Else if  new peak closer in mean_ampl to the previous true peak than the last_peak
                    elseif abs(peaks(2,i) - true_peaks(2,end)-mean_dt_peaks) < (abs(last_peaks(2)- true_peaks(2,end)-mean_dt_peaks))
                        last_peaks = peaks(:,i);
                    end
                end
                
            else
                true_peaks = [true_peaks last_peaks];
                nb_true_peaks = nb_true_peaks+1;
                mean_ampl_peaks = (last_peaks(1)/nb_true_peaks) + (((nb_true_peaks-1)/nb_true_peaks)*mean_ampl_peaks);
                mean_dt_peaks = (last_peaks(2)/nb_true_peaks) + (((nb_true_peaks-1)/nb_true_peaks)*mean_dt_peaks);
                last_peaks = peaks(:,i);
            end
            
        end
        if last_peaks(1) ~= true_peaks(1, nb_true_peaks) && last_peaks(1) > true_peaks(1, nb_true_peaks)/2 % We look if the last possible new peak is already taken into account or not
            true_peaks = [true_peaks last_peaks];
            nb_true_peaks = nb_true_peaks+1;
            mean_ampl_peaks = (last_peaks(1)/nb_true_peaks) + (((nb_true_peaks-1)/nb_true_peaks)*mean_ampl_peaks);
            mean_dt_peaks = (last_peaks(2)/nb_true_peaks) + (((nb_true_peaks-1)/nb_true_peaks)*mean_dt_peaks);
        end
        
    end
    fprintf('There are %d true peaks \n', nb_true_peaks);
    
    %Display
    figure;
    plot(time,abs(H),'b',time,threshold,'k',peaks(2,:),peaks(1,:),'.r', true_peaks(2,:),true_peaks(1,:),'.g', time,mean_ampl_peaks*ones(1,N),'c');
    legend('Hilbert', 'Threshold', 'Peaks', 'True peaks','Mean ampl');
    xlabel('time(s)');
    
    % figure;
    % plot(time,ECG,'b');
    % % http://nl.mathworks.com/help/matlab/ref/colorspec.html
    % hold on;
    % plot(peaks(2,:),ECG(peaks(2,:)*fs),'.r');
    % plot(true_peaks(2,:),ECG(true_peaks(2,:)*fs),'.g');
    % legend('ECG','Peaks', 'True peaks');
    % xlabel('time(s)');
    
    
    % Final peaks : serach in true_peaks in +- 10 samples in ECG raw
    
    real_peaks = zeros(3,length(true_peaks)); % matrix 3xn : row 1 = max value, row 2 = location in time domain [s] and row 3 = location in term of nb of samples [-]
    for i = 1:length(true_peaks)
        if(true_peaks(3,i) > delta_n && true_peaks(3,i) < (((t2-t1)*fs) + 1 -delta_n) )
            [m,ind] = max(ECG((true_peaks(3,i)-delta_n):(true_peaks(3,i)+delta_n))); % search max between x-delta_n and x + delta_n
            real_peaks(1,i) = m;
            real_peaks(3,i) = ind + (true_peaks(3,i)-delta_n) - 1;
            real_peaks(2,i) = ((real_peaks(3,i)-1)/fs) + t1;
        elseif true_peaks(3,i) <= delta_n
            [m,ind] = max(ECG(1:(true_peaks(3,i)+delta_n)));  % search max between 0 and x + delta_n
            real_peaks(1,i) = m;
            real_peaks(3,i) = ind;
            real_peaks(2,i) = ((real_peaks(3,i)-1)/fs) + t1;
        elseif true_peaks(3,i) >= (((t2-t1)*fs) + 1 -delta_n)
            [m,ind] = max(ECG((true_peaks(3,i)-delta_n):end));  % search max between 0 and x + delta_n
            real_peaks(1,i) = m;
            real_peaks(3,i) = ind + (true_peaks(3,i)-delta_n) - 1;
            real_peaks(2,i) = ((real_peaks(3,i)-1)/fs) + t1;
        end    
    end
    
    figure;
    plot(time,ECG,'b');
    % http://nl.mathworks.com/help/matlab/ref/colorspec.html
    hold on;
    plot(real_peaks(2,:),real_peaks(1,:),'.g');
    legend('ECG','Real peaks');
    xlabel('time(s)');
    title('Real peaks found in the ECG');
end