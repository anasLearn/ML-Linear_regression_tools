function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); % A vector containig the values of the 
                              % cost function J, as the iterations of the
                              % gradient descent continue

for iter = 1:num_iters

    theta = theta - alpha * (1 / m) * X' * (X * theta - y);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
