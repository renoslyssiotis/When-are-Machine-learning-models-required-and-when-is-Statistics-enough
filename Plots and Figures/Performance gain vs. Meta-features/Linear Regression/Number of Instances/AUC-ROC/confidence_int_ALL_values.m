clear all
x = [10430,   2086,   4172,   6258,   8344,   1372,    274,    549,...
          823,   1098,    748,    150,    299,    449,    598,    569,...
          114,    228,    341,    455,    116,     35,     46,     70,...
           93,    106,     32,     42,     64,     85,  10000,   2000,...
         4000,   6000,   8000,    303,     61,    121,    182,    242,...
           90,     18,     36,     54,     72,    357,     71,    143,...
          214,    286,    100,     45,     50,     60,     80,   7195,...
         1439,   2878,   4317,   5756,     90,     18,     36,     54,...
           72,    768,    154,    307,    461,    614,    187,     37,...
           75,    112,    150, 245057,  49011,  98023, 147034, 196046,...
          208,     42,     83,    125,    166,   4601,    920,   1840,...
         2761,   3681,    470,     94,    188,    282,    376,    310,...
           62,    124,    186,    248,   1340,    268,    536,    804,...
         1072,  14635,   2927,   5854,   8781,  11708,  10052,   1060,...
         2120,   3180,   4240,    200,    400,    600,    800,    106,...
          106,    106,    106,    106,   1304,    154,    307,    461,...
          614,  17898,   3580,   7159,  10739,  14318];
    
y = [0.31534141,  0.26725182,  0.29048254,  0.30581133,  0.31631892,...
        0.        ,  0.01612903,  0.        ,  0.00625   ,  0.01460655,...
        0.12211456,  0.03968254,  0.10130719,  0.15846154,  0.125     ,...
        0.        ,  0.        ,  0.06335283,  0.01944444,  0.05288958,...
        0.16666667,  0.        ,  0.25      ,  0.25      ,  0.20555556,...
        0.16666667,  0.08333333,  0.        ,  0.        ,  0.16666667,...
        0.05556189,  0.03070877,  0.05556725,  0.05336842,  0.07474359,...
        0.01470588,  0.        , -0.03525641,  0.04347826, -0.025     ,...
       -0.125     ,  0.25      ,  0.        ,  0.07142857,  0.05      ,...
        0.        ,  0.        ,  0.04166667,  0.        ,  0.        ,...
        0.09375   ,  0.10714286,  0.5       ,  0.5       ,  0.        ,...
        0.04227891,  0.04461197,  0.03618685,  0.04718784,  0.04349627,...
        0.2       ,  0.5       ,  0.        ,  0.2       ,  0.13888889,...
        0.06055556,  0.20227273,  0.02777778,  0.03276353,  0.04444444,...
        0.        ,  0.16666667,  0.16666667,  0.08333333,  0.03174603,...
        0.11585358,  0.11265386,  0.11788387,  0.11627835,  0.11519371,...
        0.07451923,  0.        ,  0.26515152,  0.1       ,  0.14642857,...
        0.0352953 ,  0.07600964,  0.03943489,  0.02259735,  0.04252657,...
        0.15178571,  0.025     ,  0.07142857,  0.14583333,  0.03639847,...
        0.04645761,  0.0625    ,  0.        ,  0.02693603,  0.15808824,...
        0.0803638 ,  0.05555556,  0.07894737,  0.05178571,  0.01342593,...
        0.18558374,  0.13714859,  0.1868315 ,  0.19470261,  0.17500136,...
        0.36410094,  0.42045455,  0.45131684,  0.365699  ,  0.39302624,...
        0.07692308,  0.01666667,  0.02535302,  0.01567024,  0.1       ,...
        0.        ,  0.        ,  0.        ,  0.1       ,  0.43030303,...
        0.2       ,  0.01818182,  0.01028708,  0.01700581,  0.01672557,...
        0.0155936 ,  0.00649932,  0.00163391,  0.01555355];
xs = linspace(0,2000,2001)';

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

%Initialise hyperparameters structure
hyp = struct('mean', [], 'cov', [-1 0], 'lik',0);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x,y);
K = feval(covfunc, hyp2.cov, x);
%Obtain the predictive mean and variance of test points
[mu,s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc,x,y,xs);

%Compute the (joint) negative log probability (density): nlml
nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
%Plot the predictive mean with predictive 95% confidence bounds and training data
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)];
fill([xs; flip(xs,1)], f, [7 7 7]/8)
hold on; plot(xs, mu, 'b'); plot(x,y,'r.','LineWidth',1)
xlabel('﻿Number of instances','FontSize',14)
ylabel('﻿Performance gain of ML vs. Statistics','FontSize',14)
a = get(gca,'XTickLabel');
b = get(gca,'YTickLabel');
set(gca,'XTickLabel',a,'fontsize',14)
set(gca,'XTickLabel',b,'fontsize',14)
title('﻿Performance gain (AUC-ROC) vs. Number of instances','FontSize',14)
legend('95% predictive error bars', 'Predictive mean', 'Data', 'FontSize',12)
xlim([0 250000]);
ylim([-0.3 0.6]);
%xticks([0 200 400 600 800 1000 1200 1400 1600 1800 2000])
%xticklabels({0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000})


