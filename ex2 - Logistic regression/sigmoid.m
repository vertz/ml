function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

size_z = size(z);

for row = 1:size_z(1)
   for column = 1:size_z(2)
      g(row, column) = 1/(1 + exp(-z(row, column)));
   endfor
endfor





% =============================================================

end

