function [J,grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);
lambdas = [[0, lambda*ones(1,n-1)]];

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

Z = X*theta;
h = sigmoid(Z);
errors = (-y).*log(h) - (1 - y).*log(1 - h);
sumErrors = ones(1,m)*errors;
sumThetaSqr = lambdas*(theta.^2);
J = 1/(m) * sumErrors .+ (sumThetaSqr/(2*m));

partialDerivative = X'*(h - y) + lambdas'.*theta;

grad = partialDerivative ./ m;

% =============================================================

end
