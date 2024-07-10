
n=10; 
p=4; 
% randomly generate a dataset 
% D = randn(n,p); 

% randomly generate a data matrix
D = zeros(n,p);
D(1:end,1)  = normrnd(0,1,n,1); 
D(1:end,2)  = normrnd(0,2,n,1);
D(1:end,3)  = 2*D(1:end,2);
D(1:end,4)  = normrnd(0,3,n,1);


% Encode the feature hierarchy 
 %A1 = [1,3]; 
 %A2 = [1,2,3]; 
 %A3 = [1,2,3,4,5]; 
 % the feature hierarchy 
A1 = [1];
A2 = [1,2,3];
A3 = [1,2,3,4];
% make a cellarray of the feature hierarchy
% this stores indices of the features
Aset = {A1, A2, A3};
 % make a cellarray of the feature hierarchy to store ... 
% indices of the features 
Aset = {A1, A2, A3};

[X_flag, nflag] = FlagRep(D, Aset)


% Plain QR
disp('QR:')
[Q,~] = qr(D, "econ");
X_qr = Q(1:end,1:rank(D));

%% Measure distances
chordal_distance(X_flag, X_qr, {[1],[2],[3]})

%% correlation with last data vector
disp('correlation with last data vector')
x =  D(1:end,4)/norm( D(1:end,4));
disp('FlagRep:')
x'*X_flag(1:end,3)
disp('QR:')
x'*X_qr(1:end,3)