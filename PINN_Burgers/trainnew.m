% trainnew.m
clear;clc 

% data set
x_data = 0:0.01:pi; num = length(x_data); y_data = sin(x_data).*exp(-x_data);
% x_data = x_data';   y_data = y_data';

n1 = 300; n2 = 300;
w1 = 0*randn(n1,1); w2 = 0*randn(n2,n1); w3 = 0*randn(1,n2);
b1 = randn(n1,1); b2 = randn(n2,1); b3 = sum(y_data)/length(y_data);


% define the nn
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(1)
    ];
net = dlnetwork(layers);
net = initialize(net);

vel = [];
LearnRate = 0.2;
momentum = 0.9;

miniBatchSize = 100;
numEpochs = 500;
numObservations = numel(x_data);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

iteration = 0;
epoch = 0;

loss = 1; tol = 1e-6;

while epoch < numEpochs && loss >= tol %&& ~monitor.Stop
    epoch = epoch + 1;

    % Shuffle data.
    idx = randperm(numel(y_data));
    XTrain = x_data(idx);
    TTrain = y_data(idx);

    i = 0;
    while i < numIterationsPerEpoch %&& ~monitor.Stop
        i = i + 1;
        iteration = iteration + 1;

        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X = XTrain(idx);
        T = TTrain(idx);

        % Convert mini-batch of data to a dlarray.
        % X = dlarray(single(X),"SSCB");
        X = dlarray(X,"CB");

        % If training on a GPU, then convert data to a gpuArray.
        if canUseGPU
            X = gpuArray(X);
        end

        % Evaluate the model loss and gradients using dlfeval and the
        % modelLoss function.
        [loss,gradients] = dlfeval(@modelLoss,net,X,T);

        % Update the network parameters using the SGDM optimizer.
        [net,vel] = sgdmupdate(net,gradients,vel,LearnRate,momentum);

        fprintf('epoch = %d, iter = %d, loss = %d\n',epoch,i,loss);

        
    end
    
    figure(1); hold on
    plot(epoch,log(loss)/log(10),'b.')
    % axis([0,epoch,0,10*loss])
    pause(0.001)

end

x = x_data; y = predict(net,x');

figure(2); hold on
plot(x_data,y_data,'-k','LineWidth',1)
plot(x,y,'--b','LineWidth',1)