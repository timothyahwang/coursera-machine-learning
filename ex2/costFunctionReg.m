function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Solve for cost(J)
first_term = (-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)))/m
% extract out the first term in theta as first term is not regularized
latter_theta = theta(2:end)
second_term = lambda/(2*m)*sum(latter_theta.^2)
J = first_term + second_term

% Solve for grad
temp_grad = X'*(sigmoid(X*theta)-y)/m
latter_temp_grad = temp_grad(2:end) + (lambda/m * latter_theta)
grad = [temp_grad(1); latter_temp_grad]





% =============================================================

end
