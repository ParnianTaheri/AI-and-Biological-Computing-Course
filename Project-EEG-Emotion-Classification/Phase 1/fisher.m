function [func_J] = fisher(y_train, channel_, func)
    % Fisher Score for Feature Selection
    class1_indices = find(y_train == 1);
    class0_indices = find(y_train == 0);

    u0 = mean(func(:,channel_));
    u1 = mean(func(class1_indices, channel_));
    u2 = mean(func(class0_indices, channel_));
    S1 = var(func(class1_indices, channel_));
    S2 = var(func(class0_indices, channel_));
    Sw = S1+S2 ;
    func_J = ((u0 - u1)^2 + (u0 - u2)^2) / Sw; 
end