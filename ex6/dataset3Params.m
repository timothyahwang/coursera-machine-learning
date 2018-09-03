function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Train the model 

potential_vals = [.01, .03, .1, .3, 1, 3, 10, 30]

min_error = realmax
min_c = -1
min_sigma = -1

for c_val = potential_vals
	for sigma_val = potential_vals
		model = svmTrain(X, y, c_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val)); 
		predictions = svmPredict(model, Xval)
		prediction_error = mean(double(predictions ~= yval))

		if (prediction_error < min_error)
			min_error = prediction_error
			min_c = c_val
			min_sigma = sigma_val
		end
	end
end

C = min_c
sigma = min_sigma






% =========================================================================

end