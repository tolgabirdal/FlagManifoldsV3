
n=10; 
p=5; 
% randomly generate a dataset 
D = randn(n,p); 

% Encode the feature hierarchy 
 A1 = [1,3]; 
 A2 = [1,2,3]; 
 A3 = [1,2,3,4,5]; 
 % make a cellarray of the feature hierarchy to store ... 
% indices of the features 
Aset = {A1, A2, A3};

[X, nflag] = FlagRep(D, Aset)