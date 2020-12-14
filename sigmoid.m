function s = sigmoid(z) % cette fonction applique la formule de sigmoid a une valeur donnée
    s = 1 ./ (1 + exp(-z));
end