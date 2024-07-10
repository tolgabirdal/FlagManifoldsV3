function [X, flag_type] = FlagRep(D, Aset)

[n,~] = size(D);

% close to zero param for truncating rank of SVD
eps_rank = 1e-8;

% get the number of As
[~,k] = size(Aset);

% for feature indices
Bset = {};

% first part of the flag
Bset{1} = Aset{1};
B = D(:,Bset{1});
C = B;
[U,S,~] = svd(C, "econ");
X{1} = U(1:end,1:nnz(S>eps_rank));
P = eye(n) - X{1}*X{1}';

m = zeros(k,1);
m(1) = rank(C);

% the rest of the flag
for i=2:k
    disp(i)
    Bset{i} = setdiff(Aset{i},Aset{i-1});
    B = D(:,Bset{i});
    C = P * B;
    [U,S,~] = svd(C, "econ");
    X{i} = U(1:end,1:nnz(S>eps_rank));
    P = (eye(n) - X{i}*X{i}')*P;
    m(i) = rank(C);
end

% translate to stiefel manifold representative n x n_k
% note, in this case n = p
X = cell2mat(X);

%compute the flag type (n_1,n_2,...,n_k)
flag_type = cumsum(m);