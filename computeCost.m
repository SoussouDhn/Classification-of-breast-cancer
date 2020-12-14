function [j, grad] = computeCost( theta, x, y)  % cette fonction calcule le cost et applique le gradian decent pour calculer les theta 
    m = size(x,1); 
    x = [ones(m, 1), x]; % L'ajout d'une colone des uns a X
    
    h = sigmoid(x * theta); % calcule de l'hypothese
    j = -(1/m)*sum(y.*log(h)+(1 - y).*log(1 - h)); % calcule du cost
    
    for i = 1 : size(theta, 1)  
        grad(i) = (1/m) * sum( (h - y) .* x(:, i) ); % calcule des thetas
    end
end