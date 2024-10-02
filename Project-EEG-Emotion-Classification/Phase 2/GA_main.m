%% Phase 2
clc
clear


%% Load Data
disp("----------------Load Data----------------")
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
numFeatureExtraction = 14;
num_feature = 70;
total_feature = ch * numFeatureExtraction;

%% Feature Extraction
disp("----------------Extract Features----------------")
[Feature_best, listOfChannels, Normalized_Train_Features, idx] = GA_feature_extraction(ch, x_train, y_train, Fs, train_size, num_feature);


%% Genetic Algorithm
disp("----------------Start Genetic Algorithm----------------")

populationSize = 50;
Generations = 500;
chromosomeLength = 70;

best_j = 0;
population = zeros(populationSize, chromosomeLength);


for i = 1:populationSize
    population(i, :) = randperm(num_feature, chromosomeLength);
end


for generation = 1:Generations
    
    if mod(generation, 10) == 0
        disp(["Generation: "+ generation])
        disp(["best_j: "+ best_j])
    end

    for i=1:populationSize
    selected_features = Feature_best(:,population(i,:));
    J(i) = fitness(selected_features, y_train);
    end

    [~, idx] = sort(J, 'descend');
    parent_size = round(0.1*populationSize);
    selectedParents = population(idx(1:parent_size), :);
    
    % Crossover
    crossover_size = round(size(selectedParents, 1)/2);
    crossoverPoints = randi([1, chromosomeLength], crossover_size);

    for i = 1:2:size(selectedParents, 1) - 1
        crossoverPoint = crossoverPoints((i+1)/2);
        temp = selectedParents(i, crossoverPoint+1:end);
        selectedParents(i, crossoverPoint+1:end) = selectedParents(i+1, crossoverPoint+1:end);
        selectedParents(i+1, crossoverPoint+1:end) = temp;
    end
    
    % Mutation
    mutation_rate = 0.1;
    mutation_size = mutation_rate * chromosomeLength;

    rand_col = zeros(populationSize,mutation_size);

    for i = 1:populationSize
        rand_col(i, :) = randperm(chromosomeLength, mutation_size);
    end
    

    mutationRange = 10; 
    mutationMask = randn(size(rand_col)) * mutationRange; 

    selectedParents(rand_col) = selectedParents(rand_col) + floor(mutationMask);

    selectedParents(selectedParents < 1) = 1;
    selectedParents(selectedParents > num_feature) = num_feature;

    newPopulation = [population; selectedParents];
    

    for i = 1:size(newPopulation, 1)
        selected_features = Feature_best(:,newPopulation(i, :));
        j(i) = fitness(selected_features, y_train);
    end
    
    [~, idx] = sort(j, 'descend');

    population = newPopulation(idx(1:populationSize), :);

    bestIndividual = population(1, :);
    selected_features = Feature_best(:,bestIndividual);
    j = fitness(selected_features, y_train);
    if j>best_j
        best_j = j;
        best_pop = bestIndividual;
    end
end


best_feature = Feature_best(:,best_pop).';

%% MLP
% find the best parameters
%Neurons_hidden1 = [10 12 14 16 18 20 22 24];
%Neurons_hidden1 = [20 30 40 50];
Neurons_hidden1 = [20 30 60 70 90 100 120];
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
        
                TrainX = best_feature(:,train_indices) ;
                TrainY = y_train(1,train_indices);
                ValX = best_feature(:,valid_indices) ;
                ValY = y_train(1,valid_indices) ;
                
                net_MLP = patternnet(neuron1);
                net_MLP = train(net_MLP,TrainX,TrainY);

                net_MLP.layers{1}.transferFcn = convertStringsToChars(activation_functions(i));
               
        
                predict_y = net_MLP(ValX);
                predict_y_train = net_MLP(TrainX);
                [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,predict_y_train,1) ;

                Thr1 = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2)));
        
                
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
TestFeatures = load('TestFeatures.mat').selected_features;

%%
predict_test_MLP = net_MLP(TestFeatures);
predict_test_MLP = predict_test_MLP >= Thr1;
predict_test_MLP = 2*predict_test_MLP-1;
save("Test_MLP2.mat",'predict_test_MLP')
best_labels_MLP = [best_acc, best_activation, best_neuron];
save("GA_Best_Labels_MLP",'best_labels_MLP')

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
    
            TrainX = best_feature(:,train_indices) ;
            TrainY = y_train(1,train_indices);
            ValX = best_feature(:,valid_indices) ;
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
fprintf("Parameters of the RBF model is : \n Spread = %d \n MN = %d \n", best_spread, best_Maxnumber);
fprintf("Best accuracy is: %.2f\n",acc)
GA_best_param_rbf = [acc,best_spread,best_Maxnumber];
save("GA_Best_Param_RBF",'GA_best_param_rbf')
predict_test_RBF = net(TestFeatures);
predict_test_RBF = predict_test_RBF >= Thr;
predict_test_RBF = 2*predict_test_RBF-1;
save("Test_RBF2",'predict_test_RBF')





