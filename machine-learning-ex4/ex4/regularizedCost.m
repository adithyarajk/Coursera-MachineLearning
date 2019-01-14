function J = regularizedCost(output_layer, y)
  
  J = (-1/m) * sum(sum(y .* log(output_layer) + (1-y) .* log(1-output_layer),2)) + (lambda/(2*m))*(sum(sum(Theta1(2:end).^2, 2))+sum(sum(Theta2(2:end).^2, 2)));
