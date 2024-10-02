function J_mult = fitness(selected_feature, y_train)
    class0_indices = find(y_train==0);
    class1_indices = find(y_train==1);

% Choose random features for 1500 trials and choose the best one in the end
    j_best = 0; 

    u1 = mean(selected_feature(class1_indices,:));
    S1 = (selected_feature(class1_indices,:) - u1)'*(selected_feature(class1_indices,:) - u1);
    S1 = S1/length(class1_indices);
 
    u2 = mean(selected_feature(class0_indices,:)) ;
    S2 = (selected_feature(class0_indices,:)-u2)'*(selected_feature(class0_indices,:) - u2);
    S2 = S2/length(class0_indices);

    Sw = S1 + S2 ;

    u0 = mean(selected_feature); 
    Sb1 = (u1 - u0)'*(u1 - u0) ;
    Sb2 = (u2 - u0)'*(u2 - u0) ;

    J_mult = trace(Sb1 + Sb2) / trace(Sw) ;
   

end