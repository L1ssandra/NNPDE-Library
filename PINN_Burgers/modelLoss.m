function [loss,gradients] = modelLoss(net,x,t,tbc,xic,uic,xa,xb)

xt = [x;t];
u = forward(net,xt);

% Calculate derivatives with respect to X and T.
du = dlgradient(sum(u,"all"),{x,t});
ux = du{1};
ut = du{2};

% Calculate mseF. Enforce Burger's equation.
f = ut + u.*ux;
%u0 = 0.*f;%zeros(size(f),"like",f);
msePDE = l2loss(f,0.*f);

% Calculate mseIC. Enforce initial conditions.
xtic = [xic;0.*xic];
uicpred = forward(net,xtic);
mseIC = l2loss(uicpred,uic);

% Calculate mseBC. Enforce boundary conditions.
xtbcL = [xa + 0.*tbc;tbc];
xtbcR = [xb + 0.*tbc;tbc];
ubcLpred = forward(net,xtbcL);
ubcRpred = forward(net,xtbcR);
mseBC = l2loss(ubcLpred,ubcRpred);

% Calculated loss to be minimized by combining errors.
loss = msePDE + mseIC + mseBC;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end