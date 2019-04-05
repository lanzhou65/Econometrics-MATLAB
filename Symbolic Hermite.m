clear;

syms a b c
syms x y z
syms xs ys
syms h

%Vasicek Model 
muX = a*(b-x);
sigmaX = c;

%Transformation from X to Y
fx2y = int(1/sigmaX,x); %computes the indefinite integral of 1/sigmaX with respect to the symbolic scalar variable x
fy2x = subs(finverse(fx2y),x,y); %compute the inverse of fx2y , then sub x to y
muY_X = muX/sigmaX - sym('1')/sym('2')*diff(sigmaX, x, 1);   %diff()  computes the 1st derivative of sigmaX with respect to the variable x
muY = subs(muY_X, x, fy2x); % change muY_X to muY which is corresponding to variable y

%Transformation from Y to Z
fy2z = (y-ys)/sqrt(h);

%Obtain beta
syms Ak_g beta expectation
clear beta Ak_g expectation

K = 5;
J = K+1;
for n=1:K
    Ak_g=subs(Hermite(n), z, fy2z);
    expectation=Ak_g;

    for k=1:J 
        Ak_g=muY*diff(Ak_g,y,1)+sym('1')/sym('2')*diff(Ak_g, y, 2);
        expectation=expectation + h^k/factorial(k)*Ak_g;
    end
    %Beta{n}= sym('1')/factorial(n-1) * subs(beta{n}, y, ys);
    beta{n} = sym('1')/factorial(n-1) * subs(expectation, y, ys);
end

%Obtain density function of Z
p_Z=sym('0');

for k=1:K
    p_Z=p_Z+beta{k}*Hermite(k);
end
findsym(p_Z); %returns all symbolic variables in p_z in alphabetical order, separated by commas

%Obtain denstiy function of X and Y through the two transformations
p_Z = exp(-z^2/2)/sqrt(2*pi)*p_Z;
p_Y = (h^(-1/2))*subs(p_Z, z, fy2z);
p_X = (sigmaX^(-1))*subs(p_Y, y, fx2y);
p_X = subs(p_X, ys, subs(fx2y, x, xs)); 
p_X = simplify(p_X);

disp(p_X);
pdf_X = matlabFunction(p_X); % converts the symbolic expression or function f to a MATLAB® function with handle pdf_X
disp(pdf_X);

%MLE
[raw,~,~] = xlsread('Tbond_ - Copy.xlsx');
theta0 = [0.1 0.1 1];
Xarray = raw;
Num_X = length(Xarray);

X_mean = mean(Xarray);
X_var = var(Xarray);
llfunc = @(theta)-sum(log(pdf_X(theta(1),theta(2),theta(3),1/250,Xarray,1)));
[thetahat,fval,~,~,~,fval2] = fminunc(llfunc,theta0);
sigmahat = abs(inv(fval2/Num_X));
sigmatheta = sigmahat/Num_X; 
z_a = (thetahat(1)-0.1)/sqrt(sigmatheta(1,1));
z_b = (thetahat(2)-0.1)/sqrt(sigmatheta(2,2));
z_c = (thetahat(3)-1)/sqrt(sigmatheta(3,3));
p_a = 2*(1-normcdf(abs(z_a)));
p_b = 2*(1-normcdf(abs(z_b)));
p_c = 2*(1-normcdf(abs(z_c)));

fprintf('The MLE a is: %f\n', thetahat(1));
fprintf('The MLE b is: %f\n', thetahat(2));
fprintf('The MLE c is: %f\n', thetahat(3));
fprintf('The p-value of a is: %f\n', p_a);
fprintf('The p-value of b is: %f\n', p_b);
fprintf('The p-value of c is: %f\n', p_c);
fprintf('The se of a is: %f\n', sqrt(sigmatheta(1,1)));
fprintf('The se of b is: %f\n', sqrt(sigmatheta(2,2)));
fprintf('The se of c is: %f\n', sqrt(sigmatheta(3,3)));
%% 
function [temp]=Hermite(k)
    % This function computes the Hermite Polynomial Recursively
    syms z
    H{1}=sym('1');
    for n=2:k
        H{n}=simplify(z*H{n-1}-diff(H{n-1},z));
    end
    temp=H{k};
end