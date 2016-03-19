% Function that computes the different features 

function[mean_RR, SDNN, mean_HR, mean_HRI, RMSSD, NN50, pNN50, SD1, SD2,...
    LF_power, HF_power, LFoverHF, LF_norm, HF_norm, LF_f_max, HF_f_max, dt_tot, Pxx, pxx1] ...
    = fctComputeFeatures( t1, t2, fs, dt_int, real_peaks)
    
    % Once we have found all the R peaks, we need to find the sub-interval on
    % which we will compute the diffrent features.
    time = (t1:1/fs:t2); 
    ind_start = 0;
    %dt_int = 20; % Nb of seconds in 1 interval
    nb_int = ceil( 2*((t2-t1)/dt_int) - 1 ); % Nb of intervals of dt_int time between t1 and t2
    RR_series = cell(nb_int,1); % Cell for dt of RR interval
    RR_kept_int = cell(nb_int,1); % Cell for the time of each RR interval ( = time at which each R peak occures since the second one)

    fs_RR = 8; % Sampling frequency for the evenly sampled tachogram
    RR_ev_samp = cell(nb_int,1); % Cell for evenly sampled RR_intervals
    time_ev_samp = zeros(nb_int, (dt_int*fs_RR));
    
    % We compute the raw RR
    dt_tot = real_peaks(2,2:end) - real_peaks(2,1:end-1);
    time_dt_tot = real_peaks(2,2:end);
    figure;
    subplot(2,1,1);
    plot(time_dt_tot, dt_tot, 'r');
    
    % We remove the non NN interval 
    % livre de ref page 97
    index_kept = 0.755*dt_tot(1:end-1) < dt_tot(2:end) &  dt_tot(2:end) < 1.325*dt_tot(1:end-1);
    dt_tot = dt_tot(index_kept);
    time_dt_tot = time_dt_tot(index_kept);
    fprintf('After removing the non NN, there are %d peaks \n', length(time_dt_tot));
    hold on;
    plot(time_dt_tot, dt_tot, 'c');
    
    % Remove outliers : https://physionet.org/tutorials/hrv-toolkit/  
    index_remove = dt_tot>2;
    dt_tot(index_remove) = [];
    time_dt_tot(index_remove) = [];
    index_remove = [];
    for i = 2:length(dt_tot)-1
        if i<= 20
            if i+19 <= length(dt_tot)
                mean_around =  [dt_tot(1:i-1) dt_tot(i+1:i+19)];
            else
                mean_around =  [dt_tot(1:i-1) dt_tot(i+1:end)];
            end
            mean_around = mean(mean_around);
        elseif (i >20) && (i<=length(dt_tot)-20)
            mean_around =  [dt_tot(i-20:i-1) dt_tot(i+1:i+19)];
            mean_around = mean(mean_around);
        else
            mean_around =  [dt_tot(i-20:i-1) dt_tot(i+1:end)];
            mean_around = mean(mean_around);   
        end
        if dt_tot(i) < 0.8*mean_around || dt_tot(i) > 1.2*mean_around
            index_remove = [index_remove i];    
        end
    end
    dt_tot(index_remove) = [];
    time_dt_tot(index_remove) = [];
    
    hold on;
    plot(time_dt_tot, dt_tot, 'b');
    xlabel('time (s)');
    ylabel('dt RR (s)');
    title('Effect of removing the outliers');
    
    fprintf('After removing the outliers, there are %d peaks \n', length(time_dt_tot));
    
    % Spline interpolation
    % We compute all the evenly sampled RR tachogram first to avoid problem
    % with limit condition on each interval by interpolating
    time_interp = (t1:1/fs_RR:(t2-(1/fs_RR)));
    RR_interp = spline( time_dt_tot, dt_tot , time_interp);
    subplot(2,1,2);
    plot(time_interp, RR_interp);
    xlabel('time (s)');
    ylabel('dt RR (s)');
    title('Evenly sampled tachogram');

    % Cnt for the different segment
    cnt = 1;

    % time domain parameters initialization
    mean_RR = zeros(1,nb_int);
    SDNN = zeros(1,nb_int);
    mean_HR = zeros(1,nb_int); % 60/mean_RR
    mean_HRI = zeros(1,nb_int); % mean(HRI) = mean(60/RRi)
    RMSSD = zeros(1,nb_int);
    NN50 = zeros(1,nb_int);
    pNN50 = zeros(1,nb_int);
    
    % Poincaré data 
    SD1 = zeros(1,nb_int);
    SD2 = zeros(1,nb_int);

    % PSD preparation
    multi_N = 8;
    freq_psd = 0:fs_RR/(multi_N*dt_int*fs_RR):fs_RR/2;

    LF_inf = ceil(0.04/(fs_RR/(multi_N*dt_int*fs_RR)))+1; %0.4 exclu
    LF_sup = round(0.15/(fs_RR/(multi_N*dt_int*fs_RR)));
    HF_inf = LF_sup+1;
    HF_sup = round(0.4/(fs_RR/(multi_N*dt_int*fs_RR)));

    RR_dft = cell(nb_int,1);
    RR_psd = cell(nb_int,1);

    LF_power = zeros(1,nb_int);
    HF_power = zeros(1,nb_int);
    LFoverHF = zeros(1,nb_int);
    LF_norm = zeros(1,nb_int);
    HF_norm = zeros(1,nb_int);
    LF_f_max = zeros(1,nb_int);
    HF_f_max = zeros(1,nb_int);
    while (t1+ind_start) < (t2- (dt_int/2))
        % We troncatenate the RR peaks in time for the current interval
        dt_RR = dt_tot(1,time_dt_tot >(t1 + ind_start) & time_dt_tot<(t1 + ind_start + dt_int));
        time_dt_RR = time_dt_tot(time_dt_tot >(t1 + ind_start) &time_dt_tot<(t1 + ind_start + dt_int));
        RR_series{cnt} = dt_RR;
        RR_kept_int{cnt} = time_dt_RR; % Usefull ???

        % We troncatenate the tachogram for the current interval
        time_ev_samp(cnt,:) = (t1+ind_start:1/fs_RR:t1+(ind_start+dt_int-(1/fs_RR)));
        RR_ev_samp{cnt} = RR_interp(time_interp >= (t1 + ind_start) & time_interp<(t1 + ind_start + dt_int));

        % We compute the PSD of the tachogram
        RR_dft{cnt} = fft(RR_ev_samp{cnt}, multi_N*dt_int*fs_RR);
        RR_dft{cnt} = RR_dft{cnt}(1:(0.5*multi_N*dt_int*fs_RR)+1);
        RR_psd{cnt} = (1/(fs_RR*(multi_N*dt_int*fs_RR))) * abs(RR_dft{cnt}).^2;
        RR_psd{cnt}(2:end-1) = 2*RR_psd{cnt}(2:end-1); % of size (multi_N*dt_int*fs_RR/2) + 1, no need to necesary be a power of 2 : http://www.mathworks.com/matlabcentral/answers/83042-how-important-is-it-to-use-power-of-2-when-using-fft

        % We compute the time domain parameters
        mean_RR(cnt) = mean(RR_series{cnt});
        SDNN(cnt) = std(RR_series{cnt});
        mean_HR(cnt) = 60/mean_RR(cnt);
        mean_HRI(cnt) = mean(60./RR_series{cnt});
        RMSSD(cnt) = sqrt( sum( (RR_series{cnt}(2:end) - RR_series{cnt}(1:end-1)).^2)/(length(RR_series{cnt})-1));
        NN50(cnt) = sum(abs(RR_series{cnt}(2:end)-RR_series{cnt}(1:end-1))>0.05);
        pNN50(cnt) = NN50(cnt)/length(RR_series{1});
        
        % Poinaré parameters 
        if cnt == 8
            figure;
            plot(RR_series{cnt}(1:end-1), RR_series{cnt}(2:end),'.');
            xlabel('RR_{n} (s)');
            ylabel('RR_{n+1} (s)');
            title('Poincaré plot for the first minute interval');
        end
        
        SD1(cnt) = (1/sqrt(2))*std(RR_series{cnt}(1:end-1), RR_series{cnt}(2:end));
        SD2(cnt) = sqrt( (2*SDNN(cnt)^2) - SD1(cnt)^2);
        
        % Histogram 
        if cnt ==8
            figure;
            h = histogram(RR_series{cnt});
            h.BinWidth = 0.008;
            %h
            title('Histogram');
        end

        % Frequency domain parameters
        LF_power(cnt) = fs_RR/(multi_N*dt_int*fs_RR)*trapz(RR_psd{cnt}(LF_inf:LF_sup));
        HF_power(cnt) = fs_RR/(multi_N*dt_int*fs_RR)*trapz(RR_psd{cnt}(HF_inf:HF_sup));
        LFoverHF(cnt) = LF_power(cnt)/HF_power(cnt);
        LF_norm(cnt) = LF_power(cnt)/(LF_power(cnt) + HF_power(cnt));
        HF_norm(cnt) = HF_power(cnt)/(LF_power(cnt) + HF_power(cnt));
        [~,LF_f_max(cnt)] = max(RR_psd{cnt}(LF_inf:LF_sup));
        LF_f_max(cnt) = freq_psd(LF_inf+LF_f_max(cnt));
        [~,HF_f_max(cnt)] = max(RR_psd{cnt}(HF_inf:HF_sup));
        HF_f_max(cnt) = freq_psd(HF_inf+HF_f_max(cnt));
        
        if cnt == 8
            figure;
            plot(freq_psd(LF_inf:HF_sup),RR_psd{cnt}(LF_inf:HF_sup));
            ylim([0 0.18]);
            xlabel('Frequency');
            ylabel('Power (ms^2/Hz)');
            title('PSD');
            
            % http://www.physionet.org/physiotools/lomb/lomb.html
            [Pxx,F]=lomb([RR_kept_int{cnt}; RR_series{cnt} ].');
            figure;
            plot(F,Pxx);
            grid on;
            title('Lomb physionet');
            
            [pxx1,f1] = plomb(RR_series{cnt},RR_kept_int{cnt});
            figure;
            plot(f1,pxx1);
            grid on;
            title('Lomb matlab');
            
            figure;
            plot(f1,pxx1.^2/(f1(2)-f1(1)));
            xlim([0.05 0.4])
            title('Try to understantd');
            
            %[pxx,f] = pyulear(dt_RR,4,5000,1);
            %subplot(1,2,2);
            %plot(f,pxx);
            %plomb(dt_RR,RR_kept(2:end),0.4);
        end

        ind_start = ind_start+ (0.5*dt_int);
        cnt = cnt + 1;
    end
    
end