function [adcData1] = Multi_readAdcData(fileFullPath, DeviceID , Start_frameIdx, End_frameIdx, numSamplePerChirp1, numChirpPerLoop1, numLoops, numRXPerDevice)
Expected_Num_Samples_Total = numSamplePerChirp1*numChirpPerLoop1*numLoops*numRXPerDevice*2;
adcData1 = readBinFile2(fileFullPath, DeviceID , Start_frameIdx, numSamplePerChirp1,numChirpPerLoop1, numLoops, numRXPerDevice, Expected_Num_Samples_Total);
end