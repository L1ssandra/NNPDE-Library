% init_data.m

%----------------
xa = 0;
xb = 2*pi;
u0 = @(x) 0.5 + sin(x);
bcL = 1;
bcR = 1;
tend = 2.2;
%----------------

hx = (xb - xa)/Nx;
dt = tend/Nt;

X = xa:hx:xb; T = 0:dt:tend;
Nx1 = Nx + 1; Nt1 = Nt + 1;

[tdata,xdata] = meshgrid(T,X);

xdata1 = reshape(xdata,1,numel(xdata));
tdata1 = reshape(tdata,1,numel(tdata));

tdatabc = [];

xdataic = [];

for i = 1:length(xdata1)
    
    if tdata1(i) == 0
        xdataic = [xdataic,xdata1(i)];
    end

    if xdata1(i) == xa
        tdatabc = [tdatabc,tdata1(i)];
    end

end

udataic = u0(xdataic);

xdata1 = rand(1,Nint)*(xb - xa) + xa; tdata1 = rand(1,Nint)*tend;

xdata1 = dlarray(double(xdata1),"CB");
tdata1 = dlarray(double(tdata1),"CB");
xdataic = dlarray(double(xdataic),"CB");
udataic = dlarray(double(udataic),"CB");
tdatabc = dlarray(double(tdatabc),"CB");

