clc
clear
close all
load agregated.mat

agregated = agregated(randperm(size(agregated, 1)), :);
agregated1 = agregated(90000:end,:);
agregated = agregated(1:90000,:);


YTrain=categorical(agregated(:,13));
anglesTrain=ones(size(agregated(:,13)));





for i=1:size(agregated(:,1:12),1)
%     agregated(i,1:12)=(agregated(i,1:12)-min(agregated(i,1:12)))/(max(agregated(i,1:12))-min(agregated(i,1:12)));
    XTrain(:,1,1,i)=agregated(i,1:12)';
end
classNames = categories(YTrain);
numClasses = numel(classNames);
numObservations = numel(YTrain);

parameters.conv1.Weights = dlarray(initializeGaussian([5,5,1,16]));
parameters.conv1.Bias = dlarray(zeros(16,1,'single'));

parameters.batchnorm1.Offset = dlarray(zeros(16,1,'single'));
parameters.batchnorm1.Scale = dlarray(ones(16,1,'single'));
state.batchnorm1.TrainedMean = zeros(16,1,'single');
state.batchnorm1.TrainedVariance  = ones(16,1,'single');

parameters.convSkip.Weights = dlarray(initializeGaussian([1,1,16,32]));
parameters.convSkip.Bias = dlarray(zeros(32,1,'single'));

parameters.batchnormSkip.Offset = dlarray(zeros(32,1,'single'));
parameters.batchnormSkip.Scale = dlarray(ones(32,1,'single'));
state.batchnormSkip.TrainedMean = zeros(32,1,'single');
state.batchnormSkip.TrainedVariance = ones(32,1,'single');

parameters.conv2.Weights = dlarray(initializeGaussian([3,3,16,32]));
parameters.conv2.Bias = dlarray(zeros(32,1,'single'));

parameters.batchnorm2.Offset = dlarray(zeros(32,1,'single'));
parameters.batchnorm2.Scale = dlarray(ones(32,1,'single'));
state.batchnorm2.TrainedMean = zeros(32,1,'single');
state.batchnorm2.TrainedVariance  = ones(32,1,'single');

parameters.conv3.Weights = dlarray(initializeGaussian([3,3,32,32]));
parameters.conv3.Bias = dlarray(zeros(32,1,'single'));

parameters.batchnorm3.Offset = dlarray(zeros(32,1,'single'));
parameters.batchnorm3.Scale = dlarray(ones(32,1,'single'));
state.batchnorm3.TrainedMean = zeros(32,1,'single');
state.batchnorm3.TrainedVariance  = ones(32,1,'single');

parameters.fc2.Weights = dlarray(initializeGaussian([6,192]));
parameters.fc2.Bias = dlarray(zeros(numClasses,1,'single'));

parameters.fc1.Weights = dlarray(initializeGaussian([1,192]));
parameters.fc1.Bias = dlarray(zeros(1,1,'single'));

% Specify Training Options

learnRate = 0.0001;
momentum = 0.5;
numEpochs = 2000;
miniBatchSize = 2000;
plots = "training-progress";
trailingAvg = [];
trailingAvgSq = [];

numIterationsPerEpoch = floor(numObservations./miniBatchSize);
executionEnvironment = "auto";

% Train Model
if plots == "training-progress"
    figure
    lineLossTrain = animatedline;
    xlabel("Iteration")
    ylabel("Loss")
end


iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Shuffle data.
    idx = randperm(numObservations);
    XTrain = XTrain(:,:,:,idx);
    YTrain = YTrain(idx);
    anglesTrain = anglesTrain(idx);
    
    % Loop over mini-batches
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        X = XTrain(:,:,:,idx);
        
        Y1 = zeros(numClasses, miniBatchSize, 'single');
        for c = 1:numClasses
            Y1(c,YTrain(idx)==classNames(c)) = 1;
        end
        
        Y2 = anglesTrain(idx)';
        Y2 = single(Y2);
        
        % Convert mini-batch of data to dlarray.
        dlX = dlarray(X,'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function.
        [gradients,state,loss] = dlfeval(@modelGradients, dlX, Y1, Y2, parameters, state);
        
        % Update the network parameters using the Adam optimizer.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg,trailingAvgSq,iteration);
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end


% Test Model

YTest=categorical(agregated1(:,13));
anglesTest=100*rand(size(agregated1(:,13)));

for i=1:size(agregated1(:,1:12),1)
%     agregated(i,1:12)=(agregated(i,1:12)-min(agregated(i,1:12)))/(max(agregated(i,1:12))-min(agregated(i,1:12)));
    XTest(:,1,1,i)=agregated1(i,1:12)';
end

dlXTest = dlarray(XTest,'SSCB');
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlXTest = gpuArray(dlXTest);
end

doTraining = false;
[dlYPred,anglesPred] = model(dlXTest, parameters,doTraining,state);

[~,idx] = max(extractdata(dlYPred),[],1);
labelsPred = classNames(idx);
accuracy = mean(labelsPred==YTest)

angleRMSE = sqrt(mean((extractdata(anglesPred) - anglesTest').^2))













function [dlY1,dlY2,state] = model(dlX,parameters,doTraining,state)

% Convolution
W = parameters.conv1.Weights;
B = parameters.conv1.Bias;
dlY = dlconv(dlX,W,B,'Padding',2);

% Batch normalization, ReLU
Offset = parameters.batchnorm1.Offset;
Scale = parameters.batchnorm1.Scale;
trainedMean = state.batchnorm1.TrainedMean;
trainedVariance = state.batchnorm1.TrainedVariance;

if doTraining
    [dlY,trainedMean,trainedVariance] = batchnorm(dlY,Offset,Scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnorm1.TrainedMean = trainedMean;
    state.batchnorm1.TrainedVariance = trainedVariance;
else
    dlY = batchnorm(dlY,Offset,Scale,trainedMean,trainedVariance);
end
dlY = relu(dlY);

% Convolution, batch normalization (Skip connection)
W = parameters.convSkip.Weights;
B = parameters.convSkip.Bias;
dlYSkip = dlconv(dlY,W,B,'Stride',2);

Offset = parameters.batchnormSkip.Offset;
Scale = parameters.batchnormSkip.Scale;
trainedMean = state.batchnormSkip.TrainedMean;
trainedVariance = state.batchnormSkip.TrainedVariance;

if doTraining
    [dlYSkip,trainedMean,trainedVariance] = batchnorm(dlYSkip,Offset,Scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnormSkip.TrainedMean = trainedMean;
    state.batchnormSkip.TrainedVariance = trainedVariance;
else
    dlYSkip = batchnorm(dlYSkip,Offset,Scale,trainedMean,trainedVariance);
end

% Convolution
W = parameters.conv2.Weights;
B = parameters.conv2.Bias;
dlY = dlconv(dlY,W,B,'Padding',1,'Stride',2);

% Batch normalization, ReLU
Offset = parameters.batchnorm2.Offset;
Scale = parameters.batchnorm2.Scale;
trainedMean = state.batchnorm2.TrainedMean;
trainedVariance = state.batchnorm2.TrainedVariance;

if doTraining
    [dlY,trainedMean,trainedVariance] = batchnorm(dlY,Offset,Scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnorm2.TrainedMean = trainedMean;
    state.batchnorm2.TrainedVariance = trainedVariance;
else
    dlY = batchnorm(dlY,Offset,Scale,trainedMean,trainedVariance);
end
dlY = relu(dlY);

% Convolution
W = parameters.conv3.Weights;
B = parameters.conv3.Bias;
dlY = dlconv(dlY,W,B,'Padding',1);

% Batch normalization
Offset = parameters.batchnorm3.Offset;
Scale = parameters.batchnorm3.Scale;
trainedMean = state.batchnorm3.TrainedMean;
trainedVariance = state.batchnorm3.TrainedVariance;

if doTraining
    [dlY,trainedMean,trainedVariance] = batchnorm(dlY,Offset,Scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnorm3.TrainedMean = trainedMean;
    state.batchnorm3.TrainedVariance = trainedVariance;
else
    dlY = batchnorm(dlY,Offset,Scale,trainedMean,trainedVariance);
end

% Addition, ReLU
dlY = dlYSkip + dlY;
dlY = relu(dlY);

% Fully connect (angles)
W = parameters.fc1.Weights;
B = parameters.fc1.Bias;
dlY2 = fullyconnect(dlY,W,B);

% Fully connect, softmax (labels)
W = parameters.fc2.Weights;
B = parameters.fc2.Bias;
dlY1 = fullyconnect(dlY,W,B);
dlY1 = softmax(dlY1);

end






function [gradients,state,loss] = modelGradients(dlX,T1,T2,parameters,state)

doTraining = true;
[dlY1,dlY2,state] = model(dlX,parameters,doTraining,state);

lossLabels = crossentropy(dlY1,T1);
lossAngles = mse(dlY2,T2);

loss = lossLabels + 0.1*lossAngles;
gradients = dlgradient(loss,parameters);

end
function parameter = initializeGaussian(sz)
parameter = randn(sz,'single').*0.01;
end