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
c = [0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
sigma1 = [0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
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


best_pred_error = inf;
pred_error = zeros(size(c), size(sigma1));
for i=1:size(c),
  for j = 1:size(sigma1),
    current_C = c(i);
    current_sigma = sigma1(j);
    model= svmTrain(X, y, current_C, @(x1, x2) gaussianKernel(x1, x2, current_sigma));
    predictions = svmPredict(model, Xval);
    pred_error(i,j) = mean(double(predictions ~= yval));
    if pred_error(i,j) < best_pred_error,
      best_pred_error = pred_error(i,j);
      C = c(i);
      sigma = sigma1(j);
    end;
  end;
end;

% =========================================================================

end
