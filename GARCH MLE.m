%% Matlab HW_2 
%Lan Zhou
%I used rng('shuffle') to test multiple times and the simulation results
%are pretty similar. I kept rng(1) to be able to reproduce. 
%L(theta) = 1415.330673;
%p_values are 0.848514 and 0.957547, which are significatly greater than
%0.05 or 0.1; plus value 1 falls within the 
%ci_calculated_1[0.849314 1.123923] and value 0.6 falls within the 
%ci_calculated_2[0.548903 0.648395],we cannot reject the null hypothesis.
%%
clear;
n = 1000;
rng(1);
shock_array = normrnd(0, 1, n);
theta_0 = [1,1];
[alpha, beta] = deal(1, 0.6);

%create 1000 autocorrelated Xi data point  
x_arr = zeros(1, n);
x_arr(1)= 2;
for i = 1 : (n-1)
    x_arr(i+1) = alpha + beta * x_arr(i) + shock_array(i);
end

%create objective function which is the negative of the likelyhood function
%calculate L(theta), MLE(thetas)
object_func = @(theta) -llhood_func(x_arr, theta);
[theta_MLE,f_val,exitflag,output,grad,hessian] = fminunc(object_func, theta_0);
matrix = -inv(hessian*(-1/n));

%calculate p_value and ci
z_1 = abs((theta_MLE(1)- alpha )/(sqrt(matrix(1,1))/sqrt(n)));
z_2 = abs((theta_MLE(2)- beta )/(sqrt(matrix(2,2))/sqrt(n)));
p_value_1 = 2*(1-normcdf(z_1));
p_value_2 = 2*(1-normcdf(z_2));

ci_calculated_1= [(theta_MLE(1) - 1.96*(sqrt(matrix(1,1))/sqrt(n))) (theta_MLE(1) + 1.96*(sqrt(matrix(1,1))/sqrt(n)))];
ci_calculated_2= [(theta_MLE(2) - 1.96*(sqrt(matrix(2,2))/sqrt(n))) (theta_MLE(2) + 1.96*(sqrt(matrix(2,2))/sqrt(n)))];
p_value = [p_value_1, p_value_2];
ci_cal = [ci_calculated_1, ci_calculated_2];

fprintf('theta_MLE(1) is: %f, theta_MLE(2) is: %f\n', theta_MLE(1), theta_MLE(2));
fprintf('Likelyhood function value is: %f\n', f_val);
fprintf('The p value for theta_MLE(1) is: %f, and the p value for theta_MLE(2) is: %f\n', p_value_1, p_value_2);
fprintf('The 95 percent confidence interval of thetahat1 is between %f %f\n', ci_calculated_1);
fprintf('The 95 percent confidence interval of thetahat2 is between %f %f\n', ci_calculated_2);

%create thh log likelyhood function
function llh_func = llhood_func(x_array, theta)
    func_g = @(x, theta) (1/(sqrt(2*pi)))*exp(-((x(1) -(theta(1) + theta(2)* x(2)))^2)/2); %change the pdf 
    llh_func = 0;
    x_dim = size(x_array, 2); 
    for i = 2: x_dim 
        x = [x_array(i-1), x_array(i)];
        llh_func = llh_func + log(func_g(x, theta));
    end
end


