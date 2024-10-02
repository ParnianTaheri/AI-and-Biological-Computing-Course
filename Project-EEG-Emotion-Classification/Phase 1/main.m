clc
clear


%% Load Data
data=load('Project_data.mat');
channel = data.Channels;
ch = length(channel);
x_test = data.TestData;
x_train = data.TrainData;
y_train = data.TrainLabels;
y_train = y_train==1;
Fs = data.fs;
train_size = size(x_train, 3);
test_size = size(x_test, 3);
total_size = train_size + test_size;
numFeatureExtractor = 14;
num_feature = 70;
total_feature = ch * numFeatureExtractor;
%% Feature Extraction
[Feature_best, j_best,Feature_best2,j_best2, Normalized_Train_Features, listOfChannels] = feature_extraction(ch, x_train, y_train, Fs, train_size, num_feature);

%% Feature

Normalized_Train_Features_vector_noise = Normalized_Train_Features(:,Feature_best).' + 0.01*rand(length(Feature_best),train_size);
%Normalized_Train_Features_vector_noise = Normalized_Train_Features(:,Feature_best).';
%Normalized_Train_Features_vector_noise = Feature_best2.';

%% MLP
% find the best parameters
%Neurons_hidden1 = [10 12 14 16 18 20 22 24];
%Neurons_hidden1 = [20 30 40 50];
Neurons_hidden1 = [60 70 80 90 100 120];
activation_functions = ["radbas", "logsig", "purelin", "satlin", "tansig", "hardlims"];

best_param1 = zeros(6,1);
acc = zeros(6,1);
fold_size = round(train_size/5);
for i = 1:length(activation_functions)
    for neuron1 = Neurons_hidden1
            fprintf("Neuron 1: %d & ",(neuron1))
            fprintf("Activation Functions: %s\n",activation_functions(i))
            
            ACC = 0 ;
            % 5-fold cross-validation
            for k=1:5
                train_indices = [1:(k-1)*fold_size,k*fold_size+1:train_size] ;
                valid_indices = (k-1)*fold_size+1:k*fold_size ;
        
                TrainX = Normalized_Train_Features_vector_noise(:,train_indices) ;
                TrainY = y_train(1,train_indices);
                ValX = Normalized_Train_Features_vector_noise(:,valid_indices) ;
                ValY = y_train(1,valid_indices) ;
                
                net_MLP = patternnet(neuron1);
                net_MLP = train(net_MLP,TrainX,TrainY);

                net_MLP.layers{1}.transferFcn = convertStringsToChars(activation_functions(i));
               
        
                predict_y = net_MLP(ValX);
                predict_y_train = net_MLP(TrainX);
                [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,predict_y_train,1) ;

                Thr1 = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2)));
                Thr1 = 0.5;
                
                predict_y = predict_y >= Thr1 ;
              
                ACC = ACC + length(find(predict_y==ValY));
            end
        
            ACCMat = ACC/(fold_size*5)*100
            if ACCMat > acc(i)
                best_param1(i) = neuron1;
                acc(i) = ACCMat;
            end
    end
end

for i = 1:length(activation_functions)
fprintf("Best accuracy: %.2f for activation function: %s with hidden neuron: %d\n",...
    acc(i), activation_functions(i), best_param1(i))
end

%% Choose the best one
ind = find(acc == max(acc));
if length(ind)>1
    ind = intersect(find(best_param1 == min(best_param1(ind))),ind);
end
best_acc = acc(ind);
best_neuron = best_param1(ind);
best_activation = activation_functions(ind);

fprintf("\n\nBest Accuracy: %.2f, for Best Activation Function: %s, With the Minimum Hidden Neuron: %d\n",...
    best_acc, best_activation, best_neuron)

%% Test

%% Feature Extraction
sz = size(Normalized_Train_Features_vector_noise);
final_feature_size = sz(1);
selected_features = test_feature(ch, x_test, Fs, test_size, listOfChannels,final_feature_size);
save("TestFeatures.mat",'selected_features')

%%
predict_test_MLP = net_MLP(selected_features);
predict_test_MLP = predict_test_MLP >= Thr1 ;
predict_test_MLP = predict_test_MLP*2-1;
save("Test_MLP1.mat",'predict_test_MLP')
best_labels_MLP = [best_acc, best_activation, best_neuron];
save("Best_Labels_MLP",'best_labels_MLP')

%% RBF
goal = 0;
ACCMat = [];
acc = 0;

Spreads = [2, 4, 6, 8 10];
MNs = [50, 60, 80,90,100];

error_min = 10000;
best_param = [];
acc=0;
for s = 1:5
    spread = Spreads(s) ;
    for m = 1:length(MNs)
        Maxnumber = MNs(m) ;
        ACC = 0 ;
        % 5-fold cross-validation
        for k=1:5
            train_indices = [1:(k-1)*fold_size,k*fold_size+1:train_size] ;
            valid_indices = (k-1)*fold_size+1:k*fold_size ;
    
            TrainX = Normalized_Train_Features_vector_noise(:,train_indices) ;
            TrainY = y_train(1,train_indices);
            ValX = Normalized_Train_Features_vector_noise(:,valid_indices) ;
            ValY = y_train(1,valid_indices) ;

            net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber);
            predict_y = net(ValX);

            Thr = 0.5 ;
            predict_y = predict_y >= Thr;
            ACC = ACC + length(find(predict_y==ValY)) ;
        end
         ACCMat = ACC/(fold_size*5)*100
    if ACCMat>acc
        best_spread = Spreads(s);
        best_Maxnumber = MNs(m);
        acc=ACCMat;
    end
    end
end
%%
fprintf("parameters of the RBF model is : \n Spread = %d \n MN = %d \n", best_spread, best_Maxnumber);
fprintf("Best accuracy is: %.2f\n",acc)
best_param_rbf = [acc,best_spread,best_Maxnumber];
save("Best_Param_RBF",'best_param_rbf')
predict_test_RBF = net(selected_features);
predict_test_RBF = predict_test_RBF >= Thr;
predict_test_RBF = predict_test_RBF * 2 - 1;
save("Test_RBF1",'predict_test_RBF')












