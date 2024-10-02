function [Normalized_Test_Features] = test_feature(ch, x_train, Fs, test_size, listOfChannels, final_feature)
    %% Feature
    J = [];
    %func = zeros(test_size,ch);
    feature_num = 0;
    feature = listOfChannels;
    for j = 1:ch
        disp(["Calculating feature "+j])
        if feature(j) == 1
            %power_Theta
            func(:,j) = test_calculate_power(x_train,test_size,j,Fs,4,7);
   
        elseif feature(j) == 2
            %power_Alpha
            func(:,j) = test_calculate_power(x_train,test_size,j,Fs,8,12);
        elseif feature(j) == 3
            %power_Low_range_Beta
            func(:,j) = test_calculate_power(x_train,test_size,j,Fs,12,15);
        elseif feature(j) == 4
            %power_Mid_range_Beta
            func(:,j) = test_calculate_power(x_train,test_size,j,Fs,16,20);
        elseif feature(j) == 5
            %power_High_range_Beta
            func(:,j) = test_calculate_power(x_train,test_size,j,Fs,21,30);
        elseif feature(j) == 6
            %power_Gamma
            func(:,j) = test_calculate_power(x_train,test_size,j,Fs,30,100);
        elseif feature(j) == 7
            % Medfreq
            for i=1:test_size 
              func(i,j) = medfreq(squeeze(x_train(j,:,i)),Fs);
            end
        elseif feature(j) == 8
            % Meanfreq
            for i=1:test_size
              func(i,j) = meanfreq(x_train(j,:,i),Fs);
            end
        elseif feature(j) == 9
            % Var Max
            for i=1:test_size
              func(i,j) = var(squeeze(x_train(j,:,i)));
            end
        elseif feature(j) == 10
            % Max Abs
            for i=1:test_size
              func(i,j) = max(abs(squeeze(x_train(j,:,i))));
            end
        elseif feature(j) == 11
            % Kurtosis
            for i = 1:test_size
              func(i,j) = kurtosis(squeeze(x_train(j,:,i)));
            end
        elseif feature(j) == 12
            % 99 Percent Bandwidth
            for i = 1:test_size
                func(i, j) = obw(squeeze(x_train(j,:,i)));         
            end
        elseif feature(j) == 13
            % Maximum Power Frequency
            for i = 1:test_size
                x = squeeze(x_train(j,:,i));
                n = length(x);
                y = fftshift(fft(x));
                f = (-n/2:n/2-1)*(Fs/n);      
                power = abs(y).^2/n;           
                index = find(power == max(power));
                func(i,j) =  max(power);
            end
        elseif feature(j) == 14
            % Band Power Feature
            for i = 1:test_size
                func(i, j) = bandpower(squeeze(x_train(j,:,i)));            
            end
        end

    end

    for j = 1:(final_feature - ch)
        rand = randi(1, final_feature-ch);
        jom = j + 59;
        disp(["Calculating feature "+jom])
        if rand == 1
            %power_Theta
            func2(:,j) = test_calculate_power(x_train,test_size,j,Fs,4,7);
        elseif rand == 2
            %power_Alpha
            func2(:,j) = test_calculate_power(x_train,test_size,j,Fs,8,12);
        elseif rand == 3
            %power_Low_range_Beta
            func2(:,j) = test_calculate_power(x_train,test_size,j,Fs,12,15);
        elseif rand == 4
            %power_Mid_range_Beta
            func2(:,j) = test_calculate_power(x_train,test_size,j,Fs,16,20);
        elseif rand == 5
            %power_High_range_Beta
            func2(:,j) = test_calculate_power(x_train,test_size,j,Fs,21,30);
        elseif rand == 6
            %power_Gamma
            func2(:,j) = test_calculate_power(x_train,test_size,j,Fs,30,100);
        elseif rand == 7
            % Medfreq
            for i=1:test_size 
              func2(i,j) = medfreq(squeeze(x_train(j,:,i)),Fs);
            end
        elseif rand == 8
            % Meanfreq
            for i=1:test_size
              func2(i,j) = meanfreq(x_train(j,:,i),Fs);
            end
        elseif rand == 9
            % Var Max
            for i=1:test_size
              func2(i,j) = var(squeeze(x_train(j,:,i)));
            end
        elseif rand == 10
            % Max Abs
            for i=1:test_size
              func2(i,j) = max(abs(squeeze(x_train(j,:,i))));
            end
        elseif rand == 11
            % Kurtosis
            for i = 1:test_size
              func2(i,j) = kurtosis(squeeze(x_train(j,:,i)));
            end
        elseif rand == 12
            % 99 Percent Bandwidth
            for i = 1:test_size
                func2(i, j) = obw(squeeze(x_train(j,:,i)));         
            end
        elseif rand == 13
            % Maximum Power Frequency
            for i = 1:test_size
                x = squeeze(x_train(j,:,i));
                n = length(x);
                y = fftshift(fft(x));
                f = (-n/2:n/2-1)*(Fs/n);      
                power = abs(y).^2/n;           
                index = find(power == max(power));
                func2(i,j) =  max(power);
            end
        elseif rand == 14
            % Band Power Feature
            for i = 1:test_size
                func2(i, j) = bandpower(squeeze(x_train(j,:,i)));            
            end
        end

    end

    func = [func,func2];



    %% Normalize
    % Vectorize
    vector_Test_Features = reshape(func, test_size, []);
     
    % Map to [-1 1]
    [Normalized_Test_Features_1,xPS] = mapminmax(vector_Test_Features') ;
    Normalized_Test_Features = Normalized_Test_Features_1';
    
    % Between [0 1]
    Normalized_Test_Features = (Normalized_Test_Features+1)./2;
    Normalized_Test_Features = Normalized_Test_Features.';
end