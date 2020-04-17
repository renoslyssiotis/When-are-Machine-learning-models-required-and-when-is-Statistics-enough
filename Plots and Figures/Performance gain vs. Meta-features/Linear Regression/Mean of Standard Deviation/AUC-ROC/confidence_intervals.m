clear all
x = [1.00004794, 1.00023978, 1.00011987, 1.00007991, 1.00005993,...
       1.00036463, 1.00182983, 1.00091199, 1.00060809, 1.00045568,...
       1.00066912, 1.00335009, 1.00167645, 1.00111545, 1.00083717,...
       1.00087989, 1.00441503, 1.00220022, 1.00146951, 1.00110072,...
       1.00433842, 1.01459931, 1.01105006, 1.00722031, 1.00542009,...
       1.00475062, 1.01600102, 1.01212165, 1.00790526, 1.00593477,...
       1.00005   , 1.00025009, 1.00023593, 1.00008334, 1.00006251,...
       1.00165426, 1.0082989 , 1.00415802, 1.00275863, 1.00207254,...
       1.00560228, 1.02899151, 1.01418511, 1.00938988, 1.00701763,...
       1.00140351, 1.00711753, 1.00351495, 1.00234467, 1.00175285,...
       1.00503782, 1.01129979, 1.01015254, 1.00843897, 1.00630921,...
       1.0000695 , 1.00034764, 1.00017378, 1.00011584, 1.00008688,...
       1.00560228, 1.02899151, 1.01418511, 1.00938988, 1.00701763,...
       1.00065168, 1.00326265, 1.00163265, 1.00108637, 1.00081533,...
       1.00268457, 1.01379376, 1.00673408, 1.0044944 , 1.00335009,...
       1.00000204, 1.0000102 , 1.0000051 , 1.0000034 , 1.00000255,...
       1.00241255, 1.01212165, 1.00607908, 1.00402416, 1.00302573,...
       1.00010869, 1.00054392, 1.00027185, 1.00018114, 1.00013586,...
       1.00106553, 1.00536197, 1.00267023, 1.00177778, 1.00133245,...
       1.00161682, 1.0081634 , 1.00405681, 1.00269906, 1.00202225,...
       1.00037334, 1.00187091, 1.00093414, 1.00062247, 1.00046674,...
       1.00003417, 1.00017087, 1.00008542, 1.00005695, 1.00004271,...
       1.00009435, 1.00047203, 1.00023593, 1.00015727, 1.00011795,...
       1.00250941, 1.00125235, 1.00083438, 1.00062559, 1.0081634 ,...
       1.04446594, 1.02062073, 1.01379376, 1.01015254, 1.00065253,...
       1.00326265, 1.00163265, 1.00108637, 1.00081533, 1.00002794,...
       1.00013969, 1.00006985, 1.00004656, 1.00003492, 1.00011972,...
       1.00059934, 1.00029936, 1.00019958, 1.00014964, 1.00238949,...
       1.01212165, 1.00600606, 1.00399203, 1.00298954, 1.00353983,...
       1.01835015, 1.00888906, 1.00593477, 1.00441503, 1.00003338,...
       1.00016693, 1.00008346, 1.00005563, 1.00004172, 1.00149143,...
       1.00754728, 1.00375236, 1.00248448, 1.00186393, 1.00026536,...
       1.0013289 , 1.00066379, 1.00044238, 1.00033173, 1.00060441,...
       1.00302573, 1.00151401, 1.00100756, 1.00075614, 1.00257401,...
       1.01307245, 1.00647256, 1.0043011 , 1.00322062, 1.0004779 ,...
       1.00240096, 1.00119546, 1.00079713, 1.00059719, 1.02062073,...
       1.05409255, 1.02597835, 1.00047427, 1.00237812, 1.00118694,...
       1.00079083, 1.00059294, 1.00092721, 1.00466203, 1.00232288,...
       1.00154679, 1.00115942, 1.00275863, 1.01418511, 1.0069205 ,...
       1.00461896, 1.00344235, 1.00086468, 1.00433842, 1.00216216,...
       1.00144404, 1.00108167];

y =     [0.31534141,  0.26725182,  0.29048254,  0.30581133,  0.31631892,...
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
        0.0155936 ,  0.00649932,  0.00163391,  0.01555355,  0.0938409 ,...
        0.13111888,  0.20776463,  0.07349896,  0.07339483,  0.07638889,...
        0.        ,  0.        ,  0.        ,  0.        ,  0.03125   ,...
        0.2       ,  0.1       ,  0.        ,  0.        ,  0.30265223,...
        0.19363721,  0.28137433,  0.30076042,  0.28801039,  0.015625  ,...
        0.16666667,  0.        ,  0.02272727,  0.01666667,  0.06773119,...
        0.19740778,  0.15947612,  0.14365751,  0.0440548 ,  0.03035167,...
        0.0625    ,  0.015625  ,  0.03125   , -0.00079365,  0.13636364,...
        0.16666667,  0.04166667,  0.08403361,  0.04      ,  0.        ,...
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,...
        0.        ,  0.        ,  0.01880927,  0.01724138,  0.00961538,...
        0.01720867,  0.025     ,  0.025     ,  0.025     , -0.0125    ,...
        0.10860656,  0.00641026,  0.11188811,  0.33333333,  0.375     ,...
        0.04166667,  0.16402715,  0.16402715,  0.16402715,  0.16402715,...
        0.13622291,  0.09068323];

xs = linspace(0.999, 1.035)';

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
xlabel('Mean of Standard Deviation','FontSize',14)
ylabel('﻿Performance gain of ML vs. Statistics','FontSize',14)
a = get(gca,'XTickLabel');
b = get(gca,'YTickLabel');
set(gca,'XTickLabel',a,'fontsize',14)
set(gca,'XTickLabel',b,'fontsize',14)
title('﻿Performance gain (AUC-ROC) vs. Mean of Standard Deviation','FontSize',14)
legend('95% predictive error bars', 'Predictive mean', 'Data', 'FontSize',12)
xlim([1 1.03]);
xticks(linspace(0.999, 1.035, 10));


