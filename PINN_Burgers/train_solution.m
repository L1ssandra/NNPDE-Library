% train_solution.m
accfun = dlaccelerate(@modelLoss);
lossFcn = @(net) dlfeval(accfun,net,xdata1,tdata1,tdatabc,xdataic,udataic,xa,xb);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info=["Iteration" "GradientsNorm" "StepNorm"], ...
    XLabel="Iteration");

iteration = 0;

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

uhflash = zeros(Nx1,Nt1);

for nt = 1:Nt1
    uht = predict(net,[X',T(nt)*ones(Nx1,1)]);
    uhflash(:,nt) = uht;
end