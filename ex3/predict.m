function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% Add ones to X
X = [ones(m,1) X]
hidden_layer = X*Theta1'

% Add ones to hidden_layer, don't forget that every layer must pass through a perceptron, that is the sigmoid function
hidden_layer = [ones(m, 1) sigmoid(hidden_layer)]
output_layer = hidden_layer*Theta2'

output_layer = output_layer'

[max_values p] = max(sigmoid(output_layer))

p = p'









% =========================================================================


end
