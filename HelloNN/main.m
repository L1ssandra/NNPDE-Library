% test4.m

% Polynomial
X = 0:0.01:1; X = X';
Loss = @(c) (c(1).*X.^2 + c(2).*X + c(3) - exp(X))'*(c(1).*X.^2 + c(2).*X + c(3) - exp(X));
c0 = [0,0,0];
[cstar, fval] = fminunc(Loss, c0);

f = @(x) exp(x);
g = @(x) cstar(1).*x.^2 + cstar(2).*x + cstar(3);

E1 = abs(f(X) - g(X));

% Single layer NN
X = 0:0.01:1; X = X';
sigma = @(x) 1./(1 + exp(-x));
Loss = @(c) (c(1).*sigma(c(2).*X + c(3)) + c(4).*sigma(c(5).*X + c(6)) + c(7)- exp(X))' ...
    *(c(1).*sigma(c(2).*X + c(3)) + c(4).*sigma(c(5).*X + c(6)) + c(7)- exp(X));
c0 = [0,0,0,0,0,0,0];
options = optimoptions('fminunc', 'MaxIterations', 5000, 'MaxFunctionEvaluations', 5000);
[cc, fval] = fminunc(Loss, c0, options);

f = @(x) exp(x);
g = @(x) cc(1).*sigma(cc(2).*x + cc(3)) + cc(4).*sigma(cc(5).*x + cc(6)) + cc(7);

E2 = abs(f(X) - g(X));

% Deep NN sigmoid
X = 0:0.01:1;

% The structure of NN
numLayers = 4;
numNeurons = 4;

% lbfgs
maxIterations = 1500;

layers = featureInputLayer(1);

for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        sigmoidLayer];
end

layers = [
    layers
    fullyConnectedLayer(1)];

net = dlnetwork(layers);
net = initialize(net);
net = dlupdate(@double,net);

% Loss function
X = dlarray(double(X),"CB");
accfun = dlaccelerate(@modelLoss);
lossFcn = @(net) dlfeval(accfun,net,X);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info=["Iteration" "GradientsNorm" "StepNorm"], ...
    XLabel="Iteration");

iteration = 0;

gradientTolerance = 1e-6;
stepTolerance = 1e-6;
solverState = lbfgsState;

while iteration < maxIterations && ~monitor.Stop
    iteration = iteration + 1;
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState);

    updateInfo(monitor, ...
        Iteration=iteration, ...
        GradientsNorm=solverState.GradientsNorm, ...
        StepNorm=solverState.StepNorm);

    recordMetrics(monitor,iteration,TrainingLoss=log(solverState.Loss)/log(10));

    monitor.Progress = 100 * iteration/maxIterations;

    if solverState.GradientsNorm < gradientTolerance || ...
            solverState.StepNorm < stepTolerance || ...
            solverState.LineSearchStatus == "failed"
        break
    end

end


f = @(x) exp(x);
g = @(x) forward(net,x);

E3 = abs(f(X) - g(X));



% Deep NN tanh
X = 0:0.01:1;

% The structure of NN
numLayers = 4;
numNeurons = 4;

% lbfgs
maxIterations = 1500;

layers = featureInputLayer(1);

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

% Loss function
X = dlarray(double(X),"CB");
accfun = dlaccelerate(@modelLoss);
lossFcn = @(net) dlfeval(accfun,net,X);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info=["Iteration" "GradientsNorm" "StepNorm"], ...
    XLabel="Iteration");

iteration = 0;

gradientTolerance = 1e-6;
stepTolerance = 1e-6;
solverState = lbfgsState;

while iteration < maxIterations && ~monitor.Stop
    iteration = iteration + 1;
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState);

    updateInfo(monitor, ...
        Iteration=iteration, ...
        GradientsNorm=solverState.GradientsNorm, ...
        StepNorm=solverState.StepNorm);

    recordMetrics(monitor,iteration,TrainingLoss=log(solverState.Loss)/log(10));

    monitor.Progress = 100 * iteration/maxIterations;

    if solverState.GradientsNorm < gradientTolerance || ...
            solverState.StepNorm < stepTolerance || ...
            solverState.LineSearchStatus == "failed"
        break
    end

end


f = @(x) exp(x);
g = @(x) forward(net,x);

E4 = abs(f(X) - g(X));



Xnew = 0:0.01:1;
E3 = extractdata(E3);
E4 = extractdata(E4);

figure(1); 
semilogy(Xnew ,E1,'r-','linewidth',1);hold on;
semilogy(Xnew ,E2,'b-','linewidth',1);hold on;
semilogy(Xnew ,E3,'c-','linewidth',1);hold on;
semilogy(Xnew ,E4,'k-','linewidth',1);hold on;
legend('poly','single','deep-sigmoid','deep-tanh');
