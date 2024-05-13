clear;clc

% ADJUSTBLE PARAMETERS
windowSize = 50;
imgSize = [28 28];

% FIX PARAMETERS
classes = categorical(1:6);

% CREATE FOLDER
parentFolder = "data_28";
for i = 1:6
    filename = fullfile(parentFolder,string(classes(i)));
    if ~isfolder(filename)
        mkdir(filename);
    end
end

% LOAD SIGNAL
sig = load("agregated.mat").agregated;

%% MAIN LOOP
for i = 1:windowSize:size(sig,1)
    window = i:i+windowSize-1;
    if window(end)>size(sig,1)
        window(window>size(sig,1)) = [];
    end 
    img = [];
    for sensor = 1:12
        img = cat(3, img, scalogram(sig(window, sensor), imgSize));
    end
    className = mode(sig(window,13));
    currentFolder = (fullfile(parentFolder,string(className)));
    newName = getNewName(currentFolder,className,"mat");
    save(fullfile(currentFolder,newName),"img")
    progressBar(i,size(sig,1))
end

%% FUNCTIONS
function img = scalogram(data,imgSize)
    fb = cwtfilterbank('SignalLength',length(data),'VoicesPerOctave',12);
    cfs = abs(fb.wt(data));
    img = rescale(cfs); % between [0,1]
    % img = im2uint8(img); % not recommended for .mat (.jpg?)
    img = imresize(img, imgSize); % resize
end
% -----------------------------------
function newImgName = getNewName(currentFolder,mainName,type)
    clasDir = dir(currentFolder);
    newName = numel({clasDir.name}) - 2;
    newImgName = string(mainName)+"_"+newName+"."+string(type);
end