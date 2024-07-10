clear all

% This shows that if we have redundant information, then FlagRep removes
% redundant information automatically, whereas QR decomposition and SVD does not.


n=10;% n samples
p=4; % p features

% the feature hierarchy 
A1 = [1];
A2 = [1,2,3];
A3 = [1,2,3,4];
% make a cellarray of the feature hierarchy
% this stores indices of the features
Aset = {A1, A2, A3};

% number of trials
n_trials = 1000;


cors = zeros(100,3);

for seed=1:n_trials
    
    %set random seed
    rng(seed);
    
    % randomly generate a data matrix
    D = zeros(n,p);
    D(1:end,1)  = normrnd(0,1,n,1); %d_1
    D(1:end,2)  = normrnd(0,2,n,1); %d_2
    D(1:end,3)  = 2*D(1:end,2);     %d_3
    D(1:end,4)  = normrnd(0,3,n,1); %d_4
    
    % Generate flag representatives using...
    % FlagRep
    [X_flag, flag_type] = FlagRep(D, Aset);
    % QR
    [Q,~] = qr(D, "econ");
    X_qr = Q(1:end,1:rank(D));
    % SVD
    [U,~,~] = svd(D, "econ");
    X_svd = U(1:end,1:rank(D));

    % Normalize columns of D
    column_norms = sqrt(sum(D.^2, 1)); %compute norms
    Dnorm = bsxfun(@rdivide, D, column_norms); %divide by norms
    
    % correlation with last data vector
    % FlagRep
    flag_cor = abs(Dnorm(1:end,1)'*X_flag(1:end,1))...
              +abs(Dnorm(1:end,2)'*X_flag(1:end,2))...
              +abs(Dnorm(1:end,3)'*X_flag(1:end,2))...
              +abs(Dnorm(1:end,4)'*X_flag(1:end,3));
    cors(seed,1) = flag_cor/4;
    % QR
    qr_cor = abs(Dnorm(1:end,1)'*X_qr(1:end,1))...
            +abs(Dnorm(1:end,2)'*X_qr(1:end,2))...
            +abs(Dnorm(1:end,3)'*X_qr(1:end,2))...
            +abs(Dnorm(1:end,4)'*X_qr(1:end,3));
    cors(seed,2) = qr_cor/4;
    % SVD
    svd_cor = abs(Dnorm(1:end,1)'*X_svd(1:end,1))...
             +abs(Dnorm(1:end,2)'*X_svd(1:end,2))...
             +abs(Dnorm(1:end,3)'*X_svd(1:end,2))...
             +abs(Dnorm(1:end,4)'*X_svd(1:end,3));
    cors(seed,3) = svd_cor/4;
end

%make a pretty plot
boxplot(cors, 'Colors', [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250], 'Widths', 0.5, 'Symbol', 'o')
colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250];
% Add legend
hold on;
for i = 1:3
    plot(NaN,NaN,'Color',colors(i,:),'LineWidth',2);
end
legend({'FlagRep','QR','SVD'},'Location','NorthEast');
% Customize axes and labels
set(gca, 'FontSize', 18, 'LineWidth', 1.5);
ylabel('Correlation', 'FontSize', 18);
% Remove grid lines and set background color to white
set(gca, 'XGrid', 'off', 'YGrid', 'off', 'Color', 'white')
set(gca,'XTick',[])
% Adjust figure size and save as a high-quality image
set(gcf, 'Position', [100, 100, 800, 400]);
% Crop the output
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0 0 8 4]); % [left bottom width height]
set(gcf, 'PaperSize', [8 4]); % Same size as PaperPosition
% Save as a high-quality PDF
print(gcf, 'flagrep_vs_qr_vs_svd', '-dpdf', '-r300');

