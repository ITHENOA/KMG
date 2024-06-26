clear;clc
for folder = ["12-Sep-2022-Test-Day-1-Evening-Layout3",...
        "12-Sep-2022-Test-Day-1-Morning-Layout2",...
        "13-Sep-2022-Test-Day-2-Morning-Layout1"]

matFilesDir = dir(fullfile(folder,"*.mat"));
keyGest = ["ring","ring","index","index","middle","middle","pinky","pinky",...
    "ring","ring","thumb","thumb","ring","ring","wrist","wrist","ring","ring","thumb","thumb",...
    "ring","ring"];
gestures = ["fist","fist","index","index","middle","middle","pinky","pinky",...
    "ring","ring","thumb","thumb","tripod","tripod","wristUp","wristUp",...
    "wristUpFist","wristUpFist","wristUpThumb","wristUpThumb",...
    "wristUpTripod","wristUpTripod"];


% make folders
classes = categorical(["fist","fist0","index","index0","middle","middle0","pinky","pinky0",...
    "ring","ring0","thumb","thumb0","tripod","tripod0","wristUp","wristUp0",...
    "wristUpFist","wristUpFist0","wristUpThumb","wristUpThumb0",...
    "wristUpTripod","wristUpTripod0"]);
% classes = categorical(gestures);
parentFolder = "data_mat_28_mag2";
for i = 1:length(classes)
    filename = fullfile(parentFolder,string(classes(i)));
    if ~isfolder(filename)
        mkdir(filename);
    end
end

% pars
imgSize = [28 28];
nChannel = 16;
Magnitude = @(x) sqrt(x(:,1).^2 + x(:,2).^2 + x(:,3).^2);

% main loop
nFiles = length(matFilesDir);
for file = 1:nFiles
    gest = extractBefore(matFilesDir(file).name,",");
    keyGest = get_keyGest(gest);
    
    % currentFolder = fullfile(parentFolder,string(gestures(file)));
    [sig,idx] = findLabel(fullfile(folder,matFilesDir(file).name), keyGest);
    % [sig,idx] = findLabel(fullfile(folder,matFilesDir(file).name), keyGest(file));
    
    Image = [];
    for i = 1:length(idx)-1
        if rem(i,2)==0
            cls_name = gest+"0";
        else
            cls_name = gest;
        end
        Image = [];
        for ch = 1:nChannel
            sig_mag = Magnitude(sig(idx(i):idx(i+1)-1, 1+(ch-1)*3:ch*3));
            Image = cat(3, Image, scalogram(sig_mag, imgSize));
        end
        currentFolder = fullfile(parentFolder,cls_name);
        imgName = getNewName(currentFolder,cls_name,"mat");
        % save(fullfile(currentFolder,imgName),"Image")
    end
    % for s = 2:2:length(idx)
    %     % .jpg
    %     clasDir = dir(currentFolder);
    %     newName = numel({clasDir.name}) - 2;
    %     try
    %     for c = 1:nChannel
    %         sig_mag = Magnitude(sig(idx(s-1):idx(s),1+(c-1)*3:c*3));
    %         % .mat
    %         Image(:,:,c) = scalogram(sig_mag, imgSize);
    %         % Image(:,:,c) = scalogram(sig(idx(s-1):idx(s),c), imgSize);
    %         % .jpg
    %         % img = scalogram(sig_mag, imgSize); % ch=16
    %         % % img = scalogram(sig(idx(s-1):idx(s),c), imgSize); % ch=48
    %         % imwrite(img, fullfile(parentFolder,gestures(file),"img_"+newName+"_"+c+".jpg"))
    %     end
    %     end
    %     % .mat
    %     imgName = getNewName(currentFolder,gestures(file),"mat");
    %     save(fullfile(parentFolder,string(gestures(file)),imgName),"Image")
    % end
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
    plot(normalize(key))
    hold on 
    Magnitude = @(x) sqrt(x(:,1).^2 + x(:,2).^2 + x(:,3).^2);
    plot(normalize(Magnitude(sig(:,1:3))))
    for i = 1:length(idx)-1
        if (idx(i+1)-idx(i)) < 10
            if idx(i+1) == idx(end)
                idx(end) = [];
                warning("idx end removed")
            else
                error("vasta")
            end
        end
    end
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
function keyGest = get_keyGest(gest)
    switch gest
        case 'fist'; keyGest = "ring";
        case 'index'; keyGest = "index";
        case 'middle'; keyGest = "middle";
        case 'pinky'; keyGest = "pinky";
        case 'ring'; keyGest = "ring";
        case 'thumb'; keyGest = "thumb";
        case 'tripod'; keyGest = "ring";
        case 'wristUp'; keyGest = "wrist";
        case 'wristUp-fist'; keyGest = "ring";
        case 'wristUp-thumb'; keyGest = "thumb";
        case 'wristUp-tripod'; keyGest = "ring";
        otherwise; error('Invalid gest name')
    end
end