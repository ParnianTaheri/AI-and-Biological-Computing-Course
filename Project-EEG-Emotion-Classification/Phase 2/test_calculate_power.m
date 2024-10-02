function y = test_calculate_power(x,size,j,Fs,f1,f2)
for i=1:size 
    y(i) = bandpower(squeeze(x(j,:,i)), Fs, [f1 f2])/bandpower(squeeze(x(j,:,i)));
end
end