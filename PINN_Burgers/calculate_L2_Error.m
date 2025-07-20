% calculate_L2_Error.m

% uh = zeros(1,Nx1);

uh = predict(net,[X',tend*ones(Nx1,1)]);
N01 = Nx1;
exact_burgers;
ureal = u_exact;

uE = abs(uh' - ureal);
L1_Error = 0;
L8_Error = max(max(uE));

for i = 1:Nx
    L1_Error = L1_Error + abs(uE(i))/Nx1;
end
