clear;clc
for folder = ["12-Sep-2022-Test-Day-1-Evening-Layout3",...
        "12-Sep-2022-Test-Day-1-Morning-Layout2",...
        "13-Sep-2022-Test-Day-2-Morning-Layout1"]

matFiles = dir(fullfile(folder,"*.mat"));
keyGest = ["ring","ring","index","index","middle","middle","pinky","pinky",...
    "ring","ring","thumb","thumb","ring","ring","wrist","wrist","ring","ring","thumb","thumb",...
    "ring","ring"];
gestures = ["fist","fist","index","index","middle","middle","pinky","pinky",...
    "ring","ring","thumb","thumb","tripod","tripod","wristUp","wristUp",...
    "wristUpFist","wristUpFist","wristUpThumb","wristUpThumb",...
    "wristUpTripod","wristUpTripod"];

% make folders
classes = categorical(gestures);
parentFolder = "data_mat_128_mag";
for i = 1:length(classes)
    mkdir(fullfile(parentFolder,string(classes(i))));
end

% pars
imgSize = [128 128];
nChannel = 16;
sample = 0;
Magnitude = @(x) sqrt(x(:,1).^2 + x(:,2).^2 + x(:,3).^2);

% main loop
nFiles = length(matFiles);
for file = 1:nFiles
    currentFolder = fullfile(parentFolder,string(gestures(file)));
    [sig,idx] = findLabel(fullfile(folder,matFiles(file).name), keyGest(file));
    Image = [];
    for s = 2:2:length(idx)
        % .jpg
        clasDir = dir(currentFolder);
        newName = numel({clasDir.name}) - 2;
        try
        for c = 1:nChannel
            sig_mag = Magnitude(sig(idx(s-1):idx(s),1+(c-1)*3:c*3));
            % .mat
            Image(:,:,c) = scalogram(sig_mag, imgSize);
            % Image(:,:,c) = scalogram(sig(idx(s-1):idx(s),c), imgSize);
            % .jpg
            % img = scalogram(sig_mag, imgSize); % ch=16
            % % img = scalogram(sig(idx(s-1):idx(s),c), imgSize); % ch=48
            % imwrite(img, fullfile(parentFolder,gestures(file),"img_"+newName+"_"+c+".jpg"))
        end
        end
        % .mat
        imgName = getNewName(currentFolder,gestures(file),"mat");
        save(fullfile(parentFolder,string(gestures(file)),imgName),"Image")
    end
    progressBar(file,nFiles)
end
end
%%
% testDataset(parentFolder,[imgSize,nChannel])
%% -----------------------------------
function [sig,idx] = findLabel(fileName,gest)
    key = load(fileName).(gest);
    sig = load(fileName).sensorsDataCalibratedFiltered;
    [~, idx_max] = findpeaks(key);
    [~, idx_min] = findpeaks(-key);
    idx = [1 idx_max idx_min length(key)];
    idx = sort(idx);
end
% -----------------------------------
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
% -----------------------------------
function testDataset(parentFolder,targetSize)
    nImg = [];
    clasDir = {dir(parentFolder).name};
    classes = categorical({clasDir{3:end}});
    for i = 3:numel(clasDir)
        imgFolder = fullfile(parentFolder,clasDir{i});
        imgNames = {dir(imgFolder).name};
        nImg = cat(2,nImg,numel(imgNames));
        for j = 1:numel(imgNames)
            imgDir = fullfile(imgFolder,imgNames{i});
            img = load(imgDir).Image;
            imgSize = size(img);
            if sum(imgSize ~= targetSize)
                warning(['Size : ',imgNames{i}])
            end
            if sum(isnan(img))
                warning(['NAN : ',imgNames{i}])
            end
        end
        progressBar(i,numel(clasDir))
    end
    disp('Test Done')
    histogram('Categories',classes,'BinCounts',nImg)
end
% -----------------------------------