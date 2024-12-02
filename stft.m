LengthSTFT = 256;
dataSTFTresult = zeros(LengthSTFT.*4,size(dataMTI,2)-LengthSTFT+1);
win = hamming(LengthSTFT);
for i = 1:size(dataMTI,2)-LengthSTFT+1
    for j = 1:size(dataMTI,1)      
    dataSTFTresult(:,i) = dataSTFTresult(:,i) + (abs(fftshift(fft(dataMTI(j,i:i+LengthSTFT-1).*win',LengthSTFT.*4))))';
    end
end

dataSTFTresult=(abs(dataSTFTresult)).^.2;

figure(4)
imagesc(dataSTFTresult);
xlabel('slow-time (s)');
axis xy;
