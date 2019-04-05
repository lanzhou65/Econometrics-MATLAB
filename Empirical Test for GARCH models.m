%% Log Return Garch

%clear;
%%Load the data 
[stock] = xlsread('AMZN_daily.csv');

%create log return
log_returns = diff(log(stock)); 
    
histfit(log_returns) %can't capture entire return distribution, fat tail 
plot(log_returns) % mean reverting, we are fitting volitility model. 
res = log_returns-mean(log_returns); %to de-mean the data, so the res data has mean 0 
plot(res);
squareRets = log_returns.^2;
plot(log_returns);
T = length(log_returns);
%calculate given time series standard deviation as one period
ts = timeseries(log_returns);
tsstd = std(ts); 
sum1=0;
% y = zeros(3000, 1);
% sigma = zeros(3000,1);

for i = 1: T
    sum1 = sum1 + log_returns(i)^2;
end
y(1) = sqrt(sum1 /T);
sigma(1) = sqrt(sum1/T);


%use garch to fit as it's mean reversion, 

%% GARCH modelling good? 
%check if it's a candidate for GARCH model, correlation of squared returns
figure;
subplot(4,1,1)
autocorr(log_returns); %check autocorrelation of returns, not really 
subplot(4,1,2)
parcorr(log_returns);
subplot(4,1,3) 
autocorr(squareRets) %check autocorrelation of squareRets,yes. Past Square_returns can predict future s_returns, GARCH can be used to fcst vol
subplot(4,1,4)
parcorr(squareRets)

%% reconfirm GARCH model fit using Engle's ARCH test
[h1, p1] = archtest(res, 'Lags', [10 15 20]', 'alpha', 0.05);
disp(h1) %h all equals 1, i.e. ARCH effects seem to exist 

%% reconfirm GARCH model fit using lbqtest test 
[h2, p2] = lbqtest (squareRets, 'lags', [10 15 20]', 'alpha', 0.05)
disp(h2) %h2 all 1, shows ARCH effects seem to exist 

%% Estimate a GARCH(1,1) Model of time series Returns
model = garch('GARCHLags',1,'ARCHLags',1,'Distribution','Gaussian');
[estMdl, estParamcov1, logL1] = estimate(model,log_returns); %the new Garchfit function - estimate the parameters of the garch model; volitility model fits by maximizing log likelyhood 
condVar = infer(estMdl, log_returns); %infers the conditional variances between the estimated model and original return data
condVol_stdeviation = sqrt(condVar); %return conditional volitility, sqrt of variance;
innovation1 = log_returns- estMdl.Offset; %calculate innovation distribution 

stdInnovation = innovation1 ./ condVol_stdeviation; %compute standardized innovation, assume it's iid ~ N(0,1) normal or t-distributed
squared_stdInnovation = stdInnovation.^2; %squared std residuals 

%plot innovation and cond Vol(Inferred standard deviation). 
figure;
plot(innovation1); hold on;
plot(condVol_stdeviation); hold off;
title('innovation1 and standard deviation/conditional Vol')
legend('innovation1', 'standard deviation/conditional vol') 

%Plot and Compare the Correlation of the Standardized Innovations
figure;
subplot(3,1,1)
plot(stdInnovation);
ylabel('Std_Innovations');
title('Standardized Innovations of GARCH(1,1)');
subplot(3,1,2)
autocorr(stdInnovation); %check autocorrelation/ serial correlation of stdResiduals, not really 
title('ACF of standardized innovation for GARCH(1,1)')
subplot(3,1,3) 
autocorr(squared_stdInnovation) %no autocorrelation of squared residuals either 
title('ACF of Squared standardized innovation for GARCH(1,1)')
legend('Innovation')

%lbqtest test for residual correlation 
[H1,pValue1,Stat,CriticalValue] = lbqtest (stdInnovation, 'lags', [5,10,15], 'alpha', 0.05)
disp([H1 pValue1 Stat CriticalValue]); %h all equals 0, p_values are quite big i.e. we fail to reject null that data is autocorrelated. 
[H2,pValue2,Stat,CriticalValue] = lbqtest (squared_stdInnovation, 'lags', [5,10,15], 'alpha', 0.05)
disp([H2 pValue2 Stat CriticalValue]); %h all equals 0, p_values are quite big i.e. we fail to reject null that data is autocorrelated. 

%calculate aic bic to decide which GARCH model to use
numParam(1) = sum(any(estParamcov1));
[aic1,bic1] = aicbic(logL1, numParam(1), T)

%% choose GARCH(1,1) to forecast one-period future conditional vol

rng default; % For reproducibility
%[v,y] = simulate(estMdl,1000);
% vF1 = forecast(estMdl,1,'Y0',y); vF2 = forecast(estMdl,1);

constant = estMdl.Constant
ARCH = cell2mat(estMdl.ARCH)
GARCH = cell2mat(estMdl.GARCH)

%simulate forecast 1000 times 
for i = 1:1000
    [vol_squared] = constant+ARCH*(tsstd*randn())^2+GARCH*(tsstd^2);
end

simulate_fcst_vol = mean(sqrt(vol_squared))
Y_next = estMdl.Offset+simulate_fcst_vol*randn()

fprintf('The vol for the input data is: %f\n', tsstd);
fprintf('The estimated vol for the next period is: %f\n', simulate_fcst_vol);
fprintf('The estimated response variable for the next period is: %f\n', Y_next);

%% mean of AR, MA, ARMA, ARIMA
% fprintf('The mean/constant and variance for AR model is: %f , %f\n', AR_log_returns1.Constant, AR_log_returns1.Variance)
% fprintf('The mean/constant and variance for MA model is: %f , %f\n', MA_log_returns1.Constant, MA_log_returns1.Variance)
% fprintf('The mean/constant and variance for ARMA model is: %f , %f\n', ARMA_log_returns1.Constant, ARMA_log_returns1.Variance)
% fprintf('The mean/constant and variance for ARIMA model is: %f , %f\n', ARIMA_log_returns1.Constant, ARIMA_log_returns1.Variance)

%% Estimate a GARCH(2,1) Model of time series Returns
model = garch('Offset',NaN,'GARCHLags',2,'ARCHLags',1,'Distribution','Gaussian');
[estMdl, estParamcov2, logL2] = estimate(model,log_returns); %the new Garchfit function - estimate the parameters of the garch model; volitility model fits by maximizing log likelyhood 
condVar = infer(estMdl, log_returns); %infers the conditional variances between the estimated model and original return data
condVol_stdeviation = sqrt(condVar); %return conditional volitility, sqrt of variance;
innovation1 = log_returns- estMdl.Offset; %calculate innovation distribution 

stdInnovation = innovation1 ./ condVol_stdeviation; %compute standardized innovation, assume it's iid ~ N(0,1) normal or t-distributed
squared_stdInnovation = stdInnovation.^2; %squared std residuals 

%plot innovation and cond Vol(Inferred standard deviation). 
figure;
plot(innovation1); hold on;
plot(condVol_stdeviation); hold off;
title('innovation1 and standard deviation/conditional Vol')
legend('innovation1', 'standard deviation/conditional vol') 

%Plot and Compare the Correlation of the Standardized Innovations
figure;
subplot(3,1,1)
plot(stdInnovation);
ylabel('Std_Innovations');
title('Standardized Innovations of GARCH(2,1)');
subplot(3,1,2)
autocorr(stdInnovation); %check autocorrelation/ serial correlation of stdResiduals, not really 
title('ACF of standardized innovation for GARCH(2,1)')
subplot(3,1,3) 
autocorr(squared_stdInnovation) %no autocorrelation of squared residuals either 
title('ACF of Squared standardized innovation for GARCH(2,1)')
legend('Innovation')

%lbqtest test 
[H1,pValue1,Stat,CriticalValue] = lbqtest (stdInnovation, 'lags', [5,10,15], 'alpha', 0.05)
disp([H1 pValue1 Stat CriticalValue]); %h all equals 10, p_values are quite big i.e. we fail to reject null that data is autocorrelated. 
[H2,pValue2,Stat,CriticalValue] = lbqtest (squared_stdInnovation, 'lags', [5,10,15], 'alpha', 0.05)
disp([H2 pValue2 Stat CriticalValue]); %h all equals 10, p_values are quite big i.e. we fail to reject null that data is autocorrelated. 

%calculate aic bic to decide which GARCH model to use
numParam(2) = sum(any(estParamcov2));
[aic2,bic2] = aicbic(logL2, numParam(2), T)

%% Estimate a GARCH(1,2) Model of time series Returns

model = garch('Offset',NaN,'GARCHLags',1,'ARCHLags',2,'Distribution','Gaussian');
[estMdl, estParamcov3, logL3] = estimate(model,log_returns); %the new Garchfit function - estimate the parameters of the garch model; volitility model fits by maximizing log likelyhood 
condVar = infer(estMdl, log_returns); %infers the conditional variances between the estimated model and original return data
condVol_stdeviation = sqrt(condVar); %return conditional volitility, sqrt of variance;
innovation1 = log_returns- estMdl.Offset; %calculate innovation distribution 

stdInnovation = innovation1 ./ condVol_stdeviation; %compute standardized innovation, assume it's iid ~ N(0,1) normal or t-distributed
squared_stdInnovation = stdInnovation.^2; %squared std residuals 

%plot innovation and cond Vol(Inferred standard deviation). 
figure;
plot(innovation1); hold on;
plot(condVol_stdeviation); hold off;
title('innovation1 and standard deviation/conditional Vol')
legend('innovation1', 'standard deviation/conditional vol') 

%Plot and Compare the Correlation of the Standardized Innovations
figure;
subplot(3,1,1)
plot(stdInnovation);
ylabel('Std_Innovations');
title('Standardized Innovations of GARCH(1,2)');
subplot(3,1,2)
autocorr(stdInnovation); %check autocorrelation/ serial correlation of stdResiduals, not really 
title('ACF of standardized innovation for GARCH(1,2)')
subplot(3,1,3) 
autocorr(squared_stdInnovation) %no autocorrelation of squared residuals either 
title('ACF of Squared standardized innovation for GARCH(1,2)')
legend('Innovation')

%lbqtest test 
[H1,pValue1,Stat,CriticalValue] = lbqtest (stdInnovation, 'lags', [5,10,15], 'alpha', 0.05)
disp([H1 pValue1 Stat CriticalValue]); %h all equals 10, p_values are quite big i.e. we fail to reject null that data is autocorrelated. 
[H2,pValue2,Stat,CriticalValue] = lbqtest (squared_stdInnovation, 'lags', [5,10,15], 'alpha', 0.05)
disp([H2 pValue2 Stat CriticalValue]); %h all equals 10, p_values are quite big i.e. we fail to reject null that data is autocorrelated. 

%calculate aic bic to decide which GARCH model to use
numParam(3) = sum(any(estParamcov3));
[aic3,bic3] = aicbic(logL3, numParam(3), T)
