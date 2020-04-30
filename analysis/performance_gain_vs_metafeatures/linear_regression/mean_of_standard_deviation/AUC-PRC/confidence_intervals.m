clear all
x = [1.000047942085991,
 1.000239779406396,
 1.0001198681454673,
 1.0000799073076503,
 1.0000599286849212,
 1.000364630823421,
 1.0018298276968016,
 1.000911992893605,
 1.000608087620805,
 1.0004556846898613,
 1.000669120181929,
 1.003350093135977,
 1.0016764471115338,
 1.0011154493149852,
 1.0008371705107924,
 1.0008798945829025,
 1.0044150325050518,
 1.0022002226818816,
 1.0014695085076675,
 1.0011007157982688,
 1.0043384151638894,
 1.0145993123917847,
 1.0110500592068736,
 1.0072203103706698,
 1.0054200938997546,
 1.004750620564033,
 1.0160010160015238,
 1.0121216546949476,
 1.0079052613579393,
 1.0059347702036958,
 1.0000500037503035,
 1.0002500937890808,
 1.0001250234423782,
 1.0000833437514418,
 1.0000625058599861,
 1.0016542608495937,
 1.0082988974836116,
 1.0041580220928048,
 1.0027586259307137,
 1.0020725410834095,
 1.0056022847309867,
 1.0289915108550531,
 1.01418510567422,
 1.00938987736568,
 1.007017629956027,
 1.001403509462617,
 1.0071175275436897,
 1.0035149493261808,
 1.0023446691037723,
 1.001752849723835,
 1.0050378152592119,
 1.0112997936948633,
 1.0101525445522106,
 1.0084389681792216,
 1.0063092108532554,
 1.00006949994796,
 1.0003476447176112,
 1.000173777045364,
 1.0001158412978102,
 1.000086877199242,
 1.0056022847309865,
 1.0289915108550531,
 1.01418510567422,
 1.0093898773656798,
 1.0070176299560272,
 1.0006516781401997,
 1.0032626514091003,
 1.001632654148321,
 1.0010863664257341,
 1.0008153283050323,
 1.0026845685887569,
 1.0137937550497034,
 1.0067340828210365,
 1.0044944046678455,
 1.0033500931359765,
 1.000002040348001,
 1.0000102019475843,
 1.0000051008826285,
 1.0000034005913723,
 1.0000025504315047,
 1.002412548741483,
 1.0121216546949479,
 1.0060790833484312,
 1.0040241611281233,
 1.0030257255228332,
 1.0001086897454525,
 1.000543921715498,
 1.0002718499438528,
 1.0001811430138987,
 1.0001358603362056,
 1.001065530403503,
 1.0053619687316815,
 1.0026702317227176,
 1.0017777791811968,
 1.0013324456276578,
 1.00161681593047,
 1.0081634007555278,
 1.004056811789459,
 1.0026990602396142,
 1.0020222467570028];

y = [0.2902708259641896,
 0.1747134930181895,
 0.25696983245286087,
 0.2987530563814511,
 0.29130292699354865,
 0.04371203675551494,
 0.008926218708827482,
 0.019554030874785577,
 0.04223905723905719,
 0.013381001021450456,
 0.13127801120448176,
 0.2797619047619048,
 0.07083333333333336,
 0.05694444444444441,
 0.03662464985994407,
 0.025308947108255153,
 0.0,
 0.056321839080459846,
 0.0,
 0.022410459910459868,
 0.051141166525781934,
 0.25,
 0.0,
 0.1339285714285714,
 0.21796916533758637,
 0.00454545454545463,
 0.18095238095238098,
 0.6666666666666667,
 0.33333333333333337,
 0.19117647058823528,
 0.06662664879296315,
 0.058918801563178524,
 0.056515960602824156,
 0.06354715572407327,
 0.09204107664406402,
 0.02894019560686223,
 0.0,
 0.10197802197802197,
 0.040125840125840195,
 -0.036106750392464804,
 0.0,
 0.25,
 0.1875,
 0.17439703153988872,
 0.20952380952380956,
 0.0,
 0.0,
 0.0,
 0.125,
 0.0,
 0.45000000000000007,
 0.38888888888888895,
 0.14999999999999997,
 0.08333333333333333,
 0.0,
 0.05719119958518171,
 0.051613337642749446,
 0.06819983199863766,
 0.04439159465061937,
 0.055677745173246196,
 0.06076388888888884,
 0.33333333333333337,
 0.0535714285714286,
 0.06285072951739612,
 0.016132478632478797,
 0.04070803797205991,
 0.06576024171332084,
 -0.03235581622678396,
 0.03779732811990888,
 0.05603512782728037,
 -0.011938094394234766,
 0.10119047619047616,
 0.07365967365967363,
 0.06015037593984962,
 0.06403162055335965,
 0.05563952445732756,
 0.057515703857212275,
 0.057419011688900845,
 0.056778325022668,
 0.0544967223676307,
 0.12885154061624648,
 0.15555555555555556,
 0.14141414141414144,
 0.19000000000000006,
 0.0,
 0.04884249428406118,
 0.029534067449178858,
 0.04464502888415933,
 0.047956557110041986,
 0.06813557153845162,
 0.015403368794326217,
 0.08038277511961728,
 0.007017543859649124,
 0.01855600539811067,
 0.020175438596491235,
 0.10467297275446685,
 0.13888888888888884,
 0.36476190476190473,
 0.1611842105263157,
 0.0];

xs = linspace(0.999, 1.035)';

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

%Initialise hyperparameters structure
hyp = struct('mean', [], 'cov', [-0.1 0], 'lik',0);
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
title('﻿Performance gain (AUC-PRC) vs. Mean of Standard Deviation','FontSize',14)
legend('95% predictive error bars', 'Predictive mean', 'Data', 'FontSize',12)
xlim([1 1.03]);
xticks(linspace(0.999, 1.035, 10));

