function Q1
    close all;
    clear;
    clc;

    %Difine functions
    function output = activation_function(X, Y, W1, W2, T, beta)
        output = 1 ./ (1 + exp(-1*beta*(W1.*X + W2.*Y - T))); % Here, we use element-wise multiplication   
    end


    function update_w1(slider, event)
    W1 = slider.Value;
    f_act = activation_function(X, Y, W1, W2, T, beta);
    surf(X,Y, f_act)
    colorbar
    title(['3D Plot with Weights:' ,'W1 = ', num2str(W1), ' ,W2 = ',num2str(W2),' ,T = ',num2str(T)]);
    xlabel('X')
    ylabel('Y')
    zlabel('Acitvation Function')
    set(gca, 'Position', [0.2, 0.3, 0.6, 0.6]);
    end

    function update_w2(slider, event)
    W2 = slider.Value;
    f_act = activation_function(X, Y, W1, W2, T, beta);
    surf(X,Y, f_act)
    colorbar
    title(['3D Plot with Weights: W1 = ' num2str(W1), ' ,W2 = ',num2str(W2),' ,T = ',num2str(T)]);
    xlabel('X')
    ylabel('Y')
    zlabel('Acitvation Function')
    set(gca, 'Position', [0.2, 0.3, 0.6, 0.6]);
    end

    function update_T(slider, event)
    T = slider.Value;
    f_act = activation_function(X, Y, W1, W2, T, beta);
    surf(X,Y, f_act)
    colorbar
    title(['3D Plot with Weights: W1 = ' num2str(W1), ' ,W2 = ',num2str(W2),' ,T = ',num2str(T)]);
    xlabel('X')
    ylabel('Y')
    zlabel('Acitvation Function')
    set(gca, 'Position', [0.2, 0.3, 0.6, 0.6]);
    end

 
    %Difine variables
    x = linspace(-1, 1, 100);
    y = linspace(-1, 1, 100);
    [X,Y] = meshgrid(x,y);
    W1 = 1;
    W2 = 1;
    T = 1;
    %b
    beta = 1;
    %c
    %beta = 2000;
    %d
    %beta = 0.001;
    
   
    
    % Define slider properties
    slider1 = uicontrol('Style', 'slider', 'Min', -5, 'Max', 5, 'Value', T, ...
                      'Position', [100 15 350 20], ...
                      'Callback', @update_T);
    slider2 = uicontrol('Style', 'slider', 'Min', -5, 'Max', 5, 'Value', W1, ...
                      'Position', [100 35 350 35], ...
                      'Callback', @update_w1);
    slider3 = uicontrol('Style', 'slider', 'Min', -5, 'Max', 5, 'Value', W2, ...
                      'Position', [100 55 350 50], ...
                      'Callback', @update_w2);
    minText = uicontrol('Style', 'text', 'String', num2str(-5), ...
                   'Units', 'normalized', 'Position', [0.09 0.045 0.091 0.04]);
    minText = uicontrol('Style', 'text', 'String', num2str(-5), ...
                   'Units', 'normalized', 'Position', [0.09 0.105 0.091 0.06]);
    minText = uicontrol('Style', 'text', 'String', num2str(-5), ...
                   'Units', 'normalized', 'Position', [0.09 0.17 0.091 0.08]);

    maxText = uicontrol('Style', 'text', 'String', num2str(5), ...
                   'Units', 'normalized', 'Position', [0.8 0.045 0.05 0.04]);
    maxText = uicontrol('Style', 'text', 'String', num2str(5), ...
                   'Units', 'normalized', 'Position', [0.8 0.105 0.05 0.06]);
    maxText = uicontrol('Style', 'text', 'String', num2str(5), ...
                   'Units', 'normalized', 'Position', [0.8 0.17 0.05 0.08]);
    
    label_T = uicontrol('Style', 'text', 'String', 'T', ...
                   'Units', 'normalized', 'Position', [0.465 0.01 0.05 0.03]);
    label_W1 = uicontrol('Style', 'text', 'String', 'W1', ...
                   'Units', 'normalized', 'Position', [0.465 0.095 0.05 0.03]);
    label_W2 = uicontrol('Style', 'text', 'String', 'W2', ...
                   'Units', 'normalized', 'Position', [0.465 0.18 0.05 0.03]);
    update_T(slider1, []);
    update_w1(slider2, []);
    update_w2(slider3, []);
    
end