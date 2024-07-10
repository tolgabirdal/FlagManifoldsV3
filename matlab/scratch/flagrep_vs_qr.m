clear all

%% iniialize variables

rng(123)

n=10;% n samples
p=3; % p features

% the feature hierarchy 
A1 = [1];
A2 = [1,2,3];
% make a cellarray of the feature hierarchy
% this stores indices of the features
Aset = {A1, A2};

% make the Bs beforehand
B1 = [1];
B2 = [2,3];
b_sig = [1,2];
Bset = {B1,B2};


% randomly generate a data matrix
D  = normrnd(0,1,n,p);


%% Generate flag representatives using...

% FlagRep
disp('FlagRep:')
X_flag = FlagRep(D, Aset)


% Plain QR
disp('QR:')
[Q,~] = qr(D, "econ");
X_qr = Q(1:end,1:p)


%% Measure distances
chordal_distance(X_flag, X_qr, Bset)




