function Q2
clear
clc
close all

file_path = 'iris.csv';

% Read the CSV file
data_table = readtable(file_path);

setosa_data = data_table{1:50,1:4};
versicolor_data = data_table{51:100,1:4};
virginica_data = data_table{101:150,1:4};

%%
%2.1
figure
subplot(2,3,1)
scatter([setosa_data(:,1),versicolor_data(:,1)],[setosa_data(:,2),versicolor_data(:,2)])
xlabel("SL")
ylabel("SW")
legend
subplot(2,3,2)
scatter([setosa_data(:,1),versicolor_data(:,1)],[setosa_data(:,3),versicolor_data(:,3)])
xlabel("SL")
ylabel("PL")
legend
subplot(2,3,3)
scatter([setosa_data(:,1),versicolor_data(:,1)],[setosa_data(:,4),versicolor_data(:,4)])
xlabel("SL")
ylabel("PW")
legend
subplot(2,3,4)
scatter([setosa_data(:,2),versicolor_data(:,2)],[setosa_data(:,3),versicolor_data(:,3)])
xlabel("SW")
ylabel("PL")
legend
subplot(2,3,5)
scatter([setosa_data(:,2),versicolor_data(:,2)],[setosa_data(:,4),versicolor_data(:,4)])
xlabel("SW")
ylabel("PW")
legend
subplot(2,3,6)
scatter([setosa_data(:,3),versicolor_data(:,3)],[setosa_data(:,4),versicolor_data(:,4)])
xlabel("PL")
ylabel("PW")
legend

%%
%2.2
%  line = 0.3x1+x2=1.5 so w1 = 0.3, w2 = 1, theta = 1.5
w1 = 0.3;
w2 = 1;
theta = 1.5;

%random selection
rnd1 = randi([1 50],1,5);
rnd2 = randi([1 50],1,5);
rnd_setosa = [];
rnd_versicolor = [];
for i=rnd1
    rnd_setosa = [rnd_setosa;setosa_data(i,:)];
end
for i=rnd2
    rnd_versicolor = [rnd_versicolor;versicolor_data(i,:)];
end

%plot
figure
scatter([rnd_setosa(:,3),rnd_versicolor(:,3)],[rnd_setosa(:,4),rnd_versicolor(:,4)])
hold on
x = 0:1:50;
plot(-w1 * x + theta)
hold off
ylim([0 2])
xlabel("PL")
ylabel("PW")
legend
%%
%2.3
% Define the training set size (80% of the data)
training_set_size = round(0.8 * length(setosa_data));
setosa_train = setosa_data(1:training_set_size, [3,4]);
versicolor_train = versicolor_data(1:training_set_size, [3,4]);
setosa_test = setosa_data(training_set_size+1:end, [3,4]);
versicolor_test = versicolor_data(training_set_size+1:end, [3,4]);
% Divide the data into training and testing sets
X_train = [setosa_train;versicolor_train];
Y_train= [ones(40,1);zeros(40,1)];
X_test = [setosa_test;versicolor_test];
Y_test = [ones(10,1);zeros(10,1)];

%Difine functions
function [w_online, theta_online] = online(X_train, Y_train, w_online, theta_online, learning_rate, epochs)
for epoch = 1:epochs
    error = [0 0];
    for i = 1:length(Y_train')
        y_pred =  sum(X_train(i, :) .* w_online) >= theta_online; % Activation function (Step function)
        if y_pred == Y_train(i,:)
        else
        w_online = w_online + learning_rate * (Y_train(i,:) - y_pred) .* X_train(i, :);
        theta_online = theta_online - learning_rate * (Y_train(i,:) - y_pred);
        error = error + abs(Y_train(i,:) - y_pred);
        end
    end
end
end


function [w_batch, theta_batch] = batch(X_train, Y_train, w_batch, theta_batch, learning_rate, epochs)
w_c = [0 0]; 
theta_c = 0;   
for epoch = 1:epochs
    error = [0 0];
    for i = 1:length(Y_train')
        y_pred =  sum(X_train(i, :) .* w_batch) >= theta_batch; 
        if y_pred == Y_train(i,:)
        else
        w_c = w_c + learning_rate * (Y_train(i,:) - y_pred) .* X_train(i, :);
        theta_c = theta_c - learning_rate * (Y_train(i,:) - y_pred);
        error = error + abs(Y_train(i,:) - y_pred);
        end
    end
    theta_batch = theta_batch + theta_c;
    w_batch = w_batch + w_c;
end
end



% Online Learning with TLU
initial_online_w = [-1 -1]; 
initial_online_theta = -0.3; 

initial_batch_w = [-2 -1];
initial_batch_theta = -5;

learning_rate = 0.001;
epochs = 500;

[w_online, theta_online] = online(X_train, Y_train, initial_online_w, initial_online_theta,...
    learning_rate, epochs);

% Batch Learning with TLU

[w_batch, theta_batch] = batch(X_train, Y_train, initial_batch_w, initial_batch_theta,...
    learning_rate, epochs);

% Test the models
y_pred_online = sum((X_test .* w_online)')' >= theta_online;
accuracy_online = sum(y_pred_online == Y_test) / length(Y_test') * 100;

y_pred_batch = sum((X_test .* w_batch)')' >= theta_batch;
accuracy_batch = sum(y_pred_batch == Y_test) / length(Y_test') * 100;

% Display accuracy
disp(['Online Learning Accuracy: ' num2str(accuracy_online) '%']);
disp(['Batch Learning Accuracy: ' num2str(accuracy_batch) '%']);

%%
%2.4
%new functions to detect each w and theta in each epoch
function [w_online2, theta_online2] = online2(X_train, Y_train, w_online, theta_online, learning_rate, epochs)
w_online2 = [];
theta_online2 = [];
for epoch = 1:epochs
    error = [0 0];
    for i = 1:length(Y_train')
        y_pred =  sum(X_train(i, :) .* w_online) >= theta_online; % Activation function (Step function)
        if y_pred == Y_train(i,:)
        else
        w_online = w_online + learning_rate * (Y_train(i,:) - y_pred) .* X_train(i, :);
        theta_online = theta_online - learning_rate * (Y_train(i,:) - y_pred);
        error = error + abs(Y_train(i,:) - y_pred);
        end
    end
        w_online2 = [w_online2;w_online];
        theta_online2 = [theta_online2;theta_online];
end
end


function [w_batch2, theta_batch2] = batch2(X_train, Y_train, w_batch, theta_batch, learning_rate, epochs)
w_batch2 = [];
theta_batch2 = [];
w_c = [0 0]; 
theta_c = 0;   
for epoch = 1:epochs
    error = [0 0];
    for i = 1:length(Y_train')
        y_pred =  sum(X_train(i, :) .* w_batch) >= theta_batch; 
        if y_pred == Y_train(i,:)
        else
        w_c = w_c + learning_rate * (Y_train(i,:) - y_pred) .* X_train(i, :);
        theta_c = theta_c - learning_rate * (Y_train(i,:) - y_pred);
        error = error + abs(Y_train(i,:) - y_pred);
        end
    end
    theta_batch = theta_batch + theta_c;
    w_batch = w_batch + w_c;
    w_batch2 = [w_batch2;w_batch];
    theta_batch2 = [theta_batch2;theta_batch];
end
end

[w_online2, theta_online2] = online2(X_train, Y_train, initial_online_w, initial_online_theta,...
    learning_rate, epochs);
[w_batch2, theta_batch2] = batch2(X_train, Y_train, initial_batch_w, initial_batch_theta,...
    learning_rate, epochs);


%plot
figure
sgtitle(['initial online w = ', num2str(initial_online_w(1)), ', ', num2str(initial_online_w(2)),'\newline',...
    'initial batch w = ', num2str(initial_batch_w(1)), ', ', num2str(initial_batch_w(2)),'\newline',...
    'initial online \theta = ', num2str(initial_online_theta),'\newline',...
    'initial batch \theta = ', num2str(initial_batch_theta)])
subplot(2,2,1)
plot(1:epochs,w_batch2)
xlabel("epoches")
ylabel("w of batch method")
legend
subplot(2,2,2)
plot(1:epochs, w_online2)
xlabel("epoches")
ylabel("w of online method")
legend
subplot(2,2,3)
plot(1:epochs,theta_batch2)
xlabel("epoches")
ylabel("\theta of batch method")
subplot(2,2,4)
plot(1:epochs,theta_online2)
xlabel("epoches")
ylabel("\theta of online method")

%%
%2.5
%plot
figure
scatter([setosa_train(:,1),versicolor_train(:,1)],[setosa_train(:,2),versicolor_train(:,2)])
hold on
x = 0:1:50;
plot(-w_batch(1)/w_batch(2) * x + theta_batch/w_batch(2))
plot(-w_online(1)/w_online(2) * x + theta_online/w_online(2))
hold off
ylim([0 2])
xlabel("PL")
ylabel("PW")
legend("Setosa", "Versicolor","Batch Line", "Online line")

%%
%2.6
%plot
figure
sgtitle("Testing")
scatter([setosa_test(:,1),versicolor_test(:,1)],[setosa_test(:,2),versicolor_test(:,2)])
hold on
x = 0:1:50;
plot(-w_batch(1)/w_batch(2) * x + theta_batch/w_batch(2))
plot(-w_online(1)/w_online(2) * x + theta_online/w_online(2))
hold off
ylim([0 2])
xlabel("PL")
ylabel("PW")
legend("Setosa", "Versicolor","Batch Line", "Online line")

%%
%2.7
figure
subplot(2,2,1)
scatter3(setosa_data(:,1),setosa_data(:,2),setosa_data(:,3))
hold on
scatter3(versicolor_data(:,1),versicolor_data(:,2),versicolor_data(:,3))
scatter3(virginica_data(:,1),virginica_data(:,2),virginica_data(:,3))
hold off
xlabel("SL")
ylabel("SW")
zlabel("PL")
legend("Setosa", "versicolor", "Virginica")
subplot(2,2,2)
scatter3(setosa_data(:,1),setosa_data(:,2),setosa_data(:,4))
hold on
scatter3(versicolor_data(:,1),versicolor_data(:,2),versicolor_data(:,4))
scatter3(virginica_data(:,1),virginica_data(:,2),virginica_data(:,4))
hold off
xlabel("SL")
ylabel("SW")
zlabel("PW")
legend("Setosa", "versicolor", "Virginica")
subplot(2,2,3)
scatter3(setosa_data(:,1),setosa_data(:,3),setosa_data(:,4))
hold on
scatter3(versicolor_data(:,1),versicolor_data(:,3),versicolor_data(:,4))
scatter3(virginica_data(:,1),virginica_data(:,3),virginica_data(:,4))
hold off
xlabel("SL")
ylabel("PL")
zlabel("PW")
legend("Setosa", "versicolor", "Virginica")
subplot(2,2,4)
scatter3(setosa_data(:,2),setosa_data(:,3),setosa_data(:,4))
hold on
scatter3(versicolor_data(:,2),versicolor_data(:,3),versicolor_data(:,4))
scatter3(virginica_data(:,2),virginica_data(:,3),virginica_data(:,4))
hold off
xlabel("SW")
ylabel("PL")
zlabel("PW")
legend("Setosa", "versicolor", "Virginica")
pos = get(gcf, 'Position');
set(gcf, 'Position',pos+[-300 -200 300 200])

end
