function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
size_theta = size(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
	h = sigmoid((X(i,:)* theta));
	
	if y(i) > 0
		J = J - log(h);
	else
		J = J - log(1 - h);
	endif
endfor;

J = J / m;
J = J + (theta' * theta - theta(1)^2)*(lambda / (2 * m));

for j = 1:size_theta(1)

	for i = 1:m
		grad(j) = grad(j) + (sigmoid(X(i,:)*theta) - y(i))*X(i,j);
	endfor;

	if j > 1
		grad(j) = grad(j) + (lambda * theta(j));
	endif;
	
	grad(j) = grad(j) / m ;
	
endfor;


% =============================================================

end

