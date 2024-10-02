function [y, score] = calculate_power(x, y_train,size,ch,Fs,f1,f2)
y = zeros(size,ch);
for j=1:ch
    for i=1:size 
        y(i,j) = bandpower(squeeze(x(j,:,i)), Fs, [f1 f2])/bandpower(squeeze(x(j,:,i)));
    end
end
    score(ch) = fisher(y_train,ch, y);
end