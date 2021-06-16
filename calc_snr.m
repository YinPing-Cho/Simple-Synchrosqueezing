filelist = dir('*.wav');
allFileNames = {filelist(:).name};
for k = 1 : length(allFileNames)
    filename = allFileNames{k};
    x = audioread(filename);
    SNR = snr(x);
    disp(compose('%s SNR: %.6f dB', filename, SNR));
end
