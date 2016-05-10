% Function that computes the different features 

function[mean_RR, SDNN, mean_HR, mean_HRI, RMSSD, NN50, pNN50, SD1, SD2,...
    LF_power, HF_power, LFoverHF, LF_norm, HF_norm, LF_f_max, HF_f_max, mean_BB, SDBB, diffBB, RMSSD_BB, SDSD, ECG_dt_tot, dt_tot, time_dt_tot, time_interp_EDR,resp_interp] ...
    = fctComputeFeaturesLombResp ( t1, t2, delta_move, dt_int, real_peaks, ECG_FIR)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % First part = removing non NN intervals and outliers %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % We compute the raw RR
    dt_tot = real_peaks(2,2:end) - real_peaks(2,1:end-1);
    time_dt_tot = real_peaks(2,2:end);
    ECG_dt_tot = real_peaks(1,2:end);
    R_peaks_FIR = ECG_FIR(real_peaks(3,2:end));
    %subplot(3,1,1);
    figure
    plot(time_dt_tot, dt_tot, 'r','linewidth',2.0);
    grid;
%     ylabel('RR interval [s]');
%     xlabel('Time [s]');
%     ylim([0 4]);
%     set(gca,'fontsize',22,'fontname','Times', 'LineWidth',0.5);
%     set(gcf, 'paperunits', 'inches');
%     Lx=8; Ly=6;
%     set(gcf, 'papersize', [Lx Ly]);
%     set(gcf, 'PaperPosition', [0.01*Lx 0.01*Ly 1.05*Lx 1.02*Ly]);
%     name = strcat('C:\Users\Florent\Documents\ULG\2ème Master\Master thesis\Rapport\Images\TachoRaw');
%     print(name,'-dpdf')
    
    % We remove the non NN interval 
    % livre de ref page 97
    index_kept = 0.755*dt_tot(1:end-1) < dt_tot(2:end) &  dt_tot(2:end) < 1.325*dt_tot(1:end-1);
    dt_tot = dt_tot(index_kept);
    time_dt_tot = time_dt_tot(index_kept);
    ECG_dt_tot = ECG_dt_tot(index_kept);
    R_peaks_FIR = R_peaks_FIR(index_kept);
    fprintf('After removing the non NN, there are %d peaks \n', length(time_dt_tot));
    %subplot(3,1,2);
    %figure
    hold on;
    grid;
    plot(time_dt_tot, dt_tot, 'c','linewidth',2.0);
    grid;
%     ylabel('RR interval (NN only) [s]');
%     xlabel('Time [s]');
%     ylim([0 4]);
%     set(gca,'fontsize',22,'fontname','Times', 'LineWidth',0.5);
%     set(gcf, 'paperunits', 'inches');
%     Lx=8; Ly=6;
%     set(gcf, 'papersize', [Lx Ly]);
%     set(gcf, 'PaperPosition', [0.01*Lx 0.01*Ly 1.05*Lx 1.02*Ly]);
%     name = strcat('C:\Users\Florent\Documents\ULG\2ème Master\Master thesis\Rapport\Images\TachoNN');
%     print(name,'-dpdf')
    
    % Remove outliers : https://physionet.org/tutorials/hrv-toolkit/  
    index_remove = dt_tot>2;
    dt_tot(index_remove) = [];
    time_dt_tot(index_remove) = [];
    ECG_dt_tot(index_remove) = [];
    R_peaks_FIR(index_remove) = [];
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
    ECG_dt_tot(index_remove) = [];
    R_peaks_FIR(index_remove) = [];
    %subplot(3,1,3);
    %figure
    hold on;
    plot(time_dt_tot, dt_tot, 'b','linewidth',2.0);
    grid;
%     xlabel('Time [s]');
%     ylabel('RR interval (NN only and no outlier) [s]');
    %title('Effect of removing the outliers');
%     ylim([0 4]);
%     
%     set(gca,'fontsize',22,'fontname','Times', 'LineWidth',0.5);
%     set(gcf, 'paperunits', 'inches');
%     Lx=8; Ly=6;
%     set(gcf, 'papersize', [Lx Ly]);
%     set(gcf, 'PaperPosition', [0.01*Lx 0.01*Ly 1.05*Lx 1.02*Ly]);
%     name = strcat('C:\Users\Florent\Documents\ULG\2ème Master\Master thesis\Rapport\Images\TachoOK');
%     print(name,'-dpdf')
    
    fprintf('After removing the outliers, there are %d peaks \n', length(time_dt_tot));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Respiration computation  % 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fs_interp_EDR = 10;
    time_interp_EDR = t1:1/fs_interp_EDR:time_dt_tot(end);
    resp_interp1 = interp1(time_dt_tot,R_peaks_FIR,time_interp_EDR, 'spline'); % spline ?
    %figure;
    %plot(time_interp_EDR,resp_interp1);
    %title('Inter spline');
    
    % Filtering with a FIR filter
    order_filt_EDR = 48;
    b_EDR = fir1(order_filt_EDR, 2*0.4/fs_interp_EDR, 'low'); % High pass FIR filter with cut-off frequency of 0.4Hz
    resp_interp = filter(b_EDR,1,resp_interp1);
    delay_filt_EDR = mean(grpdelay(b_EDR));
    
    time_interp_EDR = time_interp_EDR(1:end-delay_filt_EDR);
    resp_interp(1:delay_filt_EDR) = [];
    figure
    plot(time_interp_EDR,resp_interp);
    title('Inter spline + FIR smooth');
    
    % Find the local max and min
    [Maxima,MaxIdx] = findpeaks(resp_interp);
    resp_interp_inv = 1.01*max(resp_interp) - resp_interp;
    [~,MinIdx] = findpeaks(resp_interp_inv);
    Minima = resp_interp(MinIdx);
    length(MaxIdx)
    length(MinIdx)
    
    % To have the same number of peaks and valleys
    % If a valley first
    if MaxIdx(1) > MinIdx(1)
        MinIdx = MinIdx(1:length(MaxIdx));
        dt_peak_peak = (MaxIdx(:) - MinIdx(:))/fs_interp_EDR;
    else
        MaxIdx = MaxIdx(1:length(MinIdx));
        dt_peak_peak = (MinIdx(:) - MaxIdx(:))/fs_interp_EDR;
    end
    
    % Criterion from Mazzanti et al
    index_kept = dt_peak_peak > 0.5;
    dt_peak_peak = dt_peak_peak(index_kept);
    MinIdx = MinIdx(index_kept);
    MaxIdx = MaxIdx(index_kept);
    
    length(MaxIdx)
    length(MinIdx) 
    hold on;
    plot(time_interp_EDR(MinIdx),resp_interp(MinIdx),'*g');
    hold on;
    plot(time_interp_EDR(MaxIdx),resp_interp(MaxIdx),'*r');
    
    % Paper UCL : The amplitude difference between a peak and a valley should be at least 15% of the previous and the following amplitude difference.
    table = [];
    for i = 2:length(dt_peak_peak)-1
        ampl_prev = resp_interp(MaxIdx(i-1)) - resp_interp(MinIdx(i-1));
        ampl_next = resp_interp(MaxIdx(i+1)) - resp_interp(MinIdx(i+1));
        mean_ampl = (ampl_prev+ampl_next)/2;
        ampl_curr = resp_interp(MaxIdx(i)) - resp_interp(MinIdx(i));
        if ((ampl_curr/mean_ampl) * 100) < 15
            table = [table i];
        end
    end
    MinIdx(table) = [];
    MaxIdx(table) = [];
    
    dt_BB = time_interp_EDR(MinIdx(2:end)) - time_interp_EDR(MinIdx(1:end-1));
    time_dt_BB = time_interp_EDR(MinIdx(2:end));
    index_kept = dt_BB >= 1.3;
    dt_BB = dt_BB(index_kept);
    time_dt_BB = time_dt_BB(index_kept);
    MinIdx = MinIdx(index_kept);
    MaxIdx = MaxIdx(index_kept);
    
    true_idx_resp = [MinIdx,MaxIdx];
    true_idx_resp = sort(true_idx_resp);
    %resp_interp1 = interp1(time_interp_EDR(true_idx_resp),resp_interp(true_idx_resp),time_interp_EDR, 'spline'); % spline ?
    %hold on;
    %plot(time_interp_EDR(true_idx_resp),resp_interp(true_idx_resp),'*');
    
    
%     % Other try : 
%     t_500 = t1:1/500:time_dt_tot(end);
%     EDR_500 = zeros(size(t_500));
%     EDR_500(round(time_dt_tot*500)) = R_peaks_FIR;
%     [b,a] = butter(5,2*0.4/500);
%     EDR_500 = filter(b,a,EDR_500);
%     figure;
%     %subplot(2,1,1)
%     plot(t_500, EDR_500);
%     %subplot(2,1,2)
%     %plot(time_interp_EDR,resp_interp1,'r');
%     %title('500 Hz');
    
    
    %%%%%%%%%%%%%%%%%%%
    % Nb of intervals %
    %%%%%%%%%%%%%%%%%%%
    ind_start = 0;
    %delta_move = 20;
    nb_int = ceil( (((t2-t1)-dt_int)/delta_move) +1); % General formula to compute the number of interval
    
    cnt = 1; % Cnt to know on wich segment we are
    
    RR_series = cell(nb_int,1); % Cell for dt of RR interval
    RR_kept_int = cell(nb_int,1); % Cell for the time of each RR interval ( = time at which each R peak occures since the second one)
    FIR_resp = cell(nb_int,1); % Cell for the interpolated FIR R peaks evenly sampled
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialization of all the features % 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
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
    RR_psd = cell(nb_int,1);

    LF_power = zeros(1,nb_int);
    HF_power = zeros(1,nb_int);
    LFoverHF = zeros(1,nb_int);
    LF_norm = zeros(1,nb_int);
    HF_norm = zeros(1,nb_int);
    LF_f_max = zeros(1,nb_int);
    HF_f_max = zeros(1,nb_int);
    
    % Respiratory features
    mean_BB = zeros(1,nb_int);
    SDBB = zeros(1,nb_int);
    diffBB = zeros(1,nb_int);
    RMSSD_BB = zeros(1,nb_int);
    SDSD = zeros(1,nb_int);
    %pBB1 = zeros(1,nb_int);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Computation of the features %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t_current = t1;
    while t_current <= t2-dt_int
        % We troncatenate the RR peaks in time for the current interval
        dt_RR = dt_tot(1,time_dt_tot >t_current  & time_dt_tot<(t_current + dt_int));
        time_dt_RR = time_dt_tot(time_dt_tot >t_current  & time_dt_tot<(t_current + dt_int));
        
        RR_series{cnt} = dt_RR;
        RR_kept_int{cnt} = time_dt_RR; % Usefull ???
        
        idx = (time_dt_BB >t_current  & time_dt_BB<(t_current + dt_int)); 
        FIR_resp{cnt} = dt_BB(idx);
        
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
        

        % Frequency domain parameters
        % We compute the PSD of the tachogram
        [pxx1,f1] = plomb(RR_series{cnt},RR_kept_int{cnt});
        index_f1_inf = f1 > 0.04 & f1 <= 0.15;
        index_f1_inf = find(index_f1_inf == 1);
        LF_inf = index_f1_inf(1);
        LF_sup = index_f1_inf(end);
        index_f1_sup = f1 > 0.15 & f1 < 0.4;
        index_f1_sup = find(index_f1_sup == 1);
        HF_inf = index_f1_sup(1);
        HF_sup = index_f1_sup(end);
        
        RR_psd{cnt} = pxx1;
        
        LF_power(cnt) = (f1(2)-f1(1))*trapz(RR_psd{cnt}(LF_inf:LF_sup));
        HF_power(cnt) = (f1(2)-f1(1))*trapz(RR_psd{cnt}(HF_inf:HF_sup));
        LFoverHF(cnt) = LF_power(cnt)/HF_power(cnt);
        LF_norm(cnt) = LF_power(cnt)/(LF_power(cnt) + HF_power(cnt));
        HF_norm(cnt) = HF_power(cnt)/(LF_power(cnt) + HF_power(cnt));
        [LF_f_max(cnt),I_LF] = max((RR_psd{cnt}(LF_inf:LF_sup))/(LF_power(cnt) + HF_power(cnt)));
        %LF_f_max(cnt) = f1(I_LF);
        [HF_f_max(cnt),I_HF] = max((RR_psd{cnt}(HF_inf:HF_sup))/(LF_power(cnt) + HF_power(cnt)));
        %HF_f_max(cnt) = f1(I_HF);
        
%         if cnt == 8
%             figure;
%             plot(f1(LF_inf:HF_sup),pxx1(LF_inf:HF_sup),'linewidth',2.0);
%             xlabel('Frequency [Hz]');
%             ylabel('Power [ms^2/Hz]');
%             %title('Lomb');
%             ylim([0 0.1]);
%             grid;
%              set(gca,'fontsize',24,'fontname','Times', 'LineWidth',0.5);
%              set(gcf, 'paperunits', 'inches');
%              Lx=8; Ly=6;
%              set(gcf, 'papersize', [Lx Ly]);
%              set(gcf, 'PaperPosition', [0.01*Lx 0.01*Ly 1.05*Lx 1.02*Ly]);
%              name = strcat('C:\Users\Florent\Documents\ULG\2ème Master\Master thesis\Rapport\Images\LombExample');
%              print(name,'-dpdf')
%         end
        
        % Respiratory features
        mean_BB(cnt) = mean(FIR_resp{cnt});
        SDBB(cnt) = std(FIR_resp{cnt});
        diffBB(cnt) = max(FIR_resp{cnt}(:)) - min(FIR_resp{cnt}(:));
        RMSSD_BB(cnt) = sqrt( sum( (FIR_resp{cnt}(2:end) - FIR_resp{cnt}(1:end-1)).^2)/(length(FIR_resp{cnt})-1));
        SDSD(cnt) = std(FIR_resp{cnt}(2:end) - FIR_resp{cnt}(1:end-1));
         


        t_current = t_current + delta_move;
        cnt = cnt + 1;
    end
    
end 