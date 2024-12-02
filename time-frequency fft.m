rangefft = fft(radar_data_Rx,[],1);

figure(1)
imagesc(abs(rangefft));
title('rangefft');
axis xy;                                                                   

dataRange = rangefft(10:80,:);

figure(2)
imagesc(abs(dataRange));
title('range data after truncation');
axis xy;   