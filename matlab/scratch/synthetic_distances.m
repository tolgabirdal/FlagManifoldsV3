clear all

%% iniialize variables

dists = {};
algs = {};
for seed=1:100
    rng(seed)
    
    n=100;% n samples
    p=3; % p features
    
    
    % the feature hierarchy
    A1 = [1];
    A2 = [1,2,3];
    
    % the feature hierarchy
    B1 = [1];
    B2 = [2,3];
    Bset = {B1,B2};
    
    b_sig = [1,2];
    
    
    % make a cellarray of the feature hierarchy
    % this stores indices of the features
    Aset = {A1, A2};
    
    eps = .01
    
    
    
    % randomly generate a dataset
    D = {};
    D{1} = zeros(n,p);
    D{1}(1:end,1) = normrnd(0,1,n,1);
    D{1}(1:end,3) = normrnd(0,10,n,1);
    D{1}(1:end,2) = -D{1}(1:end,3);
    D{2} = zeros(n,p);
    D{2}(1:end,1) = 2*D{1}(1:end,1);
    D{2}(1:end,3) = 2*D{1}(1:end,2);
    D{2}(1:end,2) = D{2}(1:end,3);
    D{3} = zeros(n,p);
    D{3}(1:end,1) = 3*D{1}(1:end,1);
    D{3}(1:end,3) = 3*D{1}(1:end,2);
    D{3}(1:end,2) = D{3}(1:end,3);
    % D = {};
    % D{1} = zeros(n,p);
    % D{1}(1:end,1) = normrnd(0,1,n,1);
    % D{1}(1:end,3) = normrnd(0,2,n,1);
    % D{1}(1:end,2) = D{1}(1:end,3);
    % D{2} = zeros(n,p);
    % D{2}(1:end,1) = eps*randn(n,1)+D{1}(1:end,1);
    % D{2}(1:end,3) = eps*randn(n,1)+D{1}(1:end,2);
    % D{2}(1:end,2) = D{2}(1:end,3);
    % D{3} = zeros(n,p);
    % D{3}(1:end,1) = eps*randn(n,1)+D{1}(1:end,1);
    % D{3}(1:end,3) = eps*randn(n,1)+D{1}(1:end,2);
    % D{3}(1:end,2) = D{3}(1:end,3);
    % D{1} = normrnd(0,10,n,p);
    % D{2} = normrnd(0,10,n,p);
    % D{3} = normrnd(0,10,n,p);
    
    
    %% Generate flag representatives
    Xs_flag = {};
    for j=1:3
        Xs_flag{j} = FlagRep(D{j}, Aset);
    end
    
    %% Plain SVD
    Xs_svd = {};
    for j=1:3
        [U,~,~] = svd(D{j}, "econ");
        Xs_svd{j} = U;
    end
    
    %% Plain QR
    Xs_qr = {};
    for j=1:3
        [Q,~] = qr(D{j}, "econ");
        Xs_qr{j} = Q;
    end
    
    
    %% Measure distances
    
    % Euclidean
    Deuc_raw = zeros(3,3);
    for j=1:3
        for jj=1:3
            Deuc_raw(j,jj) = norm(D{j}(1:end,[1,3]) - D{jj}(1:end,[1,3]), 'fro');
        end
    end
    Deuc_raw = Deuc_raw/max(max(Deuc_raw));
    
    Bset = {[1],[2]};
    
    disp('FlagRep')
    % Flag with flag
    Dflag = zeros(3,3);
    for j=1:3
        for jj=1:3
            Dflag(j,jj) = chordal_distance(Xs_flag{j},Xs_flag{jj},Bset);
        end
    end
    Dflag = Dflag/max(max(Dflag));
    dists{end+1} = norm(Dflag-Deuc_raw);
    algs{end+1} = 'FlagRep';
    
    
    % SVD with flag
    disp('SVD')
    Dsvd = zeros(3,3);
    for j=1:3
        for jj=1:3
            Dsvd(j,jj) = chordal_distance(Xs_svd{j},Xs_svd{jj},Bset);
        end
    end
    Dsvd = Dsvd/max(max(Dsvd));
    dists{end+1} = norm(Dsvd-Deuc_raw);
    algs{end+1} = 'SVD';
    
    % QR with flag
    disp('QR')
    Dqr = zeros(3,3);
    for j=1:3
        for jj=1:3
            Dqr(j,jj) = chordal_distance(Xs_qr{j},Xs_qr{jj},Bset);
        end
    end
    Dqr = Dqr/max(max(Dqr));
    dists{end+1} = norm(Dqr-Deuc_raw);
    algs{end+1} = 'QR';

end

boxplot(cell2mat(dists), algs)