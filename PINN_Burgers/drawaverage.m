% drawaverage.m
hold on
plot(x_exact,u_exact,'k-','LineWidth',1);
plot(X,uh,'b--','linewidth',1);
axis([X(1),X(end),min(u_exact) - 0.1,max(u_exact) + 0.1])