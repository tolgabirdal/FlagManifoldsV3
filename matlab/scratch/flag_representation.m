clear all

%% inirialize variables

% declare n samples with p features
n=10;
p=5;

% randomly generate a dataset
D = randn(n,p);

% the feature hierarchy
A1 = [1,3];
A2 = [1,2,3];
A3 = [1,2,3,4,5];
% make a cellarray of the feature hierarchy
% this stores indices of the features
Aset = {A1, A2, A3};

%% Generate a flag representative

% get the number of As
[~,k] = size(Aset);

% for feature indices
Bset = {};

% first part of the flag
Bset{1} = Aset{1};
B = D(:,Bset{1});
C = B;
P = eye(n) - C*inv(C'*C)*C';
[U,~,~] = svd(C, "econ");
X{1} = U;


% the rest of the flag
for i=2:k
    Bset{i} = setdiff(Aset{i},Aset{i-1});
    B = D(:,Bset{i});
    C = P * B;
    P = (eye(n) - C*inv(C'*C)*C')*P;
    [U,~,~] = svd(C, "econ");
    X{i} = U;

end

% translate to stiefel manifold representative n x n_k
% note, in this case n = p
X = cell2mat(X);
