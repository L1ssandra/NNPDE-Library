% set_network.m
layers = featureInputLayer(2);

for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layers = [
    layers
    fullyConnectedLayer(1)];

net = dlnetwork(layers);
net = initialize(net);
net = dlupdate(@double,net);

gradientTolerance = 1e-7;
stepTolerance = 1e-7;
solverState = lbfgsState;