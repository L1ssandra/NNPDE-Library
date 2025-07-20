% flash.m

frameMAX = 500;
figure(1)
t0 = T(end)/frameMAX;
N0 = 400;
Xf = xa:(xb - xa)/N0:xb;
N01 = N0 + 1;
for i = 1:frameMAX + 1
    tt = (i - 1)*t0;
    [~,j] = min(abs(T - tt));
    tpre = tend; Xppre = Xf;
    tend = tt; X = Xf;
    exact_burgers;
    uf = predict(net,[Xf',tend*ones(N01,1)]);
    tend = tpre; X = Xppre;
    % plot(X,uhflash(:,j),'b-',x_exact,u_exact,'k-');
    plot(x_exact,u_exact,'k-',Xf(1:2:end),uf(1:2:end),'b^','linewidth',1);
    axis([X(1),X(end),min(u_exact) - 0.1,max(u_exact) + 0.1]);
    pause(0.001);
end