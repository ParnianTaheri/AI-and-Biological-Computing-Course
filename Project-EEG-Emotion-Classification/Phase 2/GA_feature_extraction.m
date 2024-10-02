
function [Feature_best, listOfChannels, Normalized_Train_Features, I2] = GA_feature_extraction(ch, x_train, y_train, Fs, train_size, num_feature)

%% Feature
J = [];
Train_Features = zeros(train_size,ch);
feature_num = 0;
%%
%power_Theta
disp("Calculating Theta")
[Theta, Theta_J] = calculate_power(x_train,y_train,train_size,ch,Fs,4,7);
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = Theta;
J = [J;Theta_J];
%power_Alpha
disp("Calculating Alpha")
[Alpha, Alpha_J] = calculate_power(x_train,y_train,train_size,ch,Fs,8,12);
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = Alpha;
J = [J;Alpha_J];
%power_Low_range_Beta
disp("Calculating LBeta")
[LBeta, LBeta_J] = calculate_power(x_train,y_train,train_size,ch,Fs,12,15);
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = LBeta;
J = [J;LBeta_J];
%power_Mid_range_Beta
disp("Calculating MBeta")
[MBeta, MBeta_J] = calculate_power(x_train,y_train,train_size,ch,Fs,16,20);
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = MBeta;
J = [J;MBeta_J];
%power_High_range_Beta
disp("Calculating HBeta")
[HBeta, HBeta_J] = calculate_power(x_train,y_train,train_size,ch,Fs,21,30);
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = HBeta;
J = [J;HBeta_J];
%power_Gamma
disp("Calculating Gamma")
[Gamma, Gamma_J] = calculate_power(x_train,y_train,train_size,ch,Fs,30,100);
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = Gamma;
J = [J;Gamma_J];


% Medfreq
disp("Calculating Medfreq")
freq_med = zeros(train_size,ch);
for j=1:ch
    for i=1:train_size 
      freq_med(i,j) = medfreq(squeeze(x_train(j,:,i)),Fs);
    end
    freq_med_J(ch) = fisher(y_train, ch, freq_med); 
end
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = freq_med;
disp(['Fisher score for best Medfreq = ', num2str(max(freq_med_J)), ' for channel = ', num2str(find(freq_med_J == max(freq_med_J)))]);
J = [J;freq_med_J];

% Meanfreq
disp("Calculating Meanfreq")
freq_mean = zeros(squeeze(train_size),ch);
for  j=1:ch 
    for i=1:train_size
      freq_mean(i,j) = meanfreq(x_train(j,:,i),Fs);
    end
    freq_mean_J(ch) = fisher(y_train, ch, freq_mean); 
end
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = freq_mean;
disp(['Fisher score for best Meanfreq = ', num2str(max(freq_mean_J)), ' for channel = ', num2str(find(freq_mean_J == max(freq_mean_J)))]);
J = [J;freq_mean_J];

% Var Max
disp("Calculating Max Var")
variance = zeros(train_size,ch);
for  j=1:ch
    for i=1:train_size
      variance(i,j) = var(squeeze(x_train(j,:,i)));
    end
    variance_J(ch) = fisher(y_train, ch, variance); 
end
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = variance;
disp(['Fisher score for best variance = ', num2str(max(variance_J)), ' for channel = ', num2str(find(variance_J == max(variance_J)))]);
J = [J;variance_J];

% Max Abs
disp("Calculating Max Abs")
max_abs = zeros(train_size,ch);
for j=1:ch
    for i=1:train_size
      max_abs(i,j) = max(abs(squeeze(x_train(j,:,i))));
    end
    max_abs_J(ch) = fisher(y_train, ch, max_abs); 
end
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = max_abs;
disp(['Fisher score for best  Max Abs = ', num2str(max(max_abs_J)), ' for channel = ', num2str(find(max_abs_J == max(max_abs_J)))]);
J = [J;max_abs_J];

% Kurtosis
disp("Calculating Kurtosis")
kurtosis_signal = zeros(train_size,ch);
for j=1:ch
    for i=1:train_size
      kurtosis_signal(i,j) = kurtosis(squeeze(x_train(j,:,i)));
    end
     kurtosis_signal_J(ch) = fisher(y_train, ch, kurtosis_signal); 
end
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = kurtosis_signal;
disp(['Fisher score for best Kurtosis = ', num2str(max(kurtosis_signal_J)), ' for channel = ', num2str(find(kurtosis_signal_J == max(kurtosis_signal_J)))]);
J = [J;kurtosis_signal_J];


% 99 Percent Bandwidth
for j = 1:ch
    for i = 1:train_size
        bandwidth_99Percent(i, j) = obw(squeeze(x_train(j,:,i)));
    end
     bandwidth_99Percent_J(j) = fisher(y_train, j, bandwidth_99Percent); 
end
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) =  bandwidth_99Percent;
disp(['Fisher score for best 99 percent bandwidth = ', num2str(max(bandwidth_99Percent_J)), ' for channel = ', num2str(find(bandwidth_99Percent_J == max(bandwidth_99Percent_J)))]);
J = [J;bandwidth_99Percent_J];

% Maximum Power Frequency
for j = 1:ch
    for i = 1:train_size
        x = squeeze(x_train(j,:,i));
        num_feature = length(x);
        y = fftshift(fft(x));
        f = (-num_feature/2:num_feature/2-1)*(Fs/num_feature);      
        power = abs(y).^2/num_feature;           
        index = find(power == max(power));
        maxPowerFrequency(i,j) =  max(power);
    end
     maxPowerFrequency_J(j) = fisher(y_train, j, maxPowerFrequency); 
end
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = maxPowerFrequency;
disp(['Fisher score for best maximum power frequency = ', num2str(max(maxPowerFrequency_J)), ' for channel = ', num2str(find(maxPowerFrequency_J == max(maxPowerFrequency_J)))]);
J = [J;maxPowerFrequency_J];

% Band Power Feature
for j = 1:ch
    for i = 1:train_size
        Bandpower(i, j) = bandpower(squeeze(x_train(j,:,i)));
    end
    Bandpower_J(j) = fisher(y_train, j, Bandpower);
end
feature_num = feature_num + 1;
Train_Features(:,:,feature_num) = Bandpower;
disp(['Fisher score for best band power = ', num2str(max(Bandpower_J)), ' for channel = ', num2str(find(Bandpower_J == max(Bandpower_J)))]);
J = [J;Bandpower_J];

%%
numFeatureExtractor = feature_num;
total_feature = ch * numFeatureExtractor;

disp('--------------------------------------------------------------------------------------------------------------------------------------------------');
for i = 1:ch
    listOfChannels(i) = find(J(:, i) == max(J(:, i)));
    disp(['The best feature for channel', num2str(i), ' is ', ListOfFeatures(listOfChannels(i)), ', with the fisher score: ', num2str(max(J(:, i)))]);
end

%% Normalize
% Vectorize
vector_Train_Features = reshape(Train_Features, train_size, total_feature);
 
% Map to [-1 1]
[Normalized_Train_Features_1,xPS] = mapminmax(vector_Train_Features') ;
Normalized_Train_Features = Normalized_Train_Features_1';

% Between [0 1]
Normalized_Train_Features = (Normalized_Train_Features+1)./2;

%% Fisher
class0_indices = find(y_train==0);
class1_indices = find(y_train==1);

J = zeros(total_feature,1);
for i=1:total_feature
    u0 = mean(Normalized_Train_Features(:,i)) ; 
    u1 = mean(Normalized_Train_Features(class1_indices,i)) ;
    S1 = var(Normalized_Train_Features(class1_indices,i));
    u2 = mean(Normalized_Train_Features(class0_indices,i)) ;
    S2 = var(Normalized_Train_Features(class0_indices,i));
    Sw = S1+S2 ;
    J(i) = (u1-u0)^2 + (u2-u0)^2 / Sw ;
end 

[B2,I2] = maxk(J,num_feature);
j_best2 = J(I2);
Feature_best = Normalized_Train_Features(:,I2);

end
