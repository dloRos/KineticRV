dataMTI = filter(b,a,dataRange,[],2);
figure(4)
imagesc(abs(dataMTI));
title('range data after MTI');
axis xy; 