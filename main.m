% Initialisation 'Suppression de toutes les variables'
clear; close all; clc;

% Chargement de la base de données
data = load('dataR2.txt');

% Chargement des features (X's)
x = data(: , 1:9);
% Chargement des resultat (Y)
y = data(: , 10);

% Dans la bese de données R1 
%    y = 2 signifie la presence du cancer 
%    y = 1 signifie que la personne n'a pas le cancer
% Avant l'implementation de ce algorithme il faut changer les valeurs de 1et 2 vers 1 et 0 
pos = find(y == 2); % collect des indexs ou y = 2
neg = find(y == 1); % collect des indexs ou y = 1
y(pos ,1) = 1; % changer les valeurs ou y == 2 vers y = 1 'Presence du cancer'
y(neg ,1) = 0; % changer les valeurs ou y == 1 vers y = 0 'Abcence du cancer'

[a,b] = size(x); % a = nombre de lignes de x et b = nombre de colones
theta = zeros( b + 1 , 1); % Inicialisation de theta par des zeros selon le nombre des features le + 1 est pour theta(0)

% inicalisation des options de la fonction fminunc qui va calculer le meilleur theta pour notre cas
options = optimset('GradObj', 'on', 'MaxIter', 300); % Options ( Objet gradien -> on , max iterations -> 300 )
[theta, cost] = fminunc(@(thetaa)(computeCost(thetaa, x, y)), theta, options); % parameteres ( la fonction du cost , theta initial , les options )
% theta ( le theta optimal ), cost ( le cout minimale )

% Affichage de theta et Cost
display(theta);
display([' Cost = ', num2str(cost)]);

% calcule et affichage du pourcentage de precision de ce algorithme
precision = accuracy(theta, x, y);
display([' precision = ', num2str(precision),' %']); % ou bien fprintf( 'precision = %.2f \n' ,precision);

% L'ajout d'une colone des uns a X
x = [ones(size(x,1), 1), x];

cancer = x(pos ,:); % le collect des lignes de x ou y = 1
nocancer = x(neg ,:); % le collect des lignes de x ou y = 0

posH = sort(sigmoid(cancer * theta)); % calcule de hypothes pour y = 1 et l'ordonner
negH = sort(sigmoid(nocancer * theta)); % calcule de hypothes pour y = 0 et l'ordonner

% les plots
figure('name','Cost Fuction 1'); % titre de la figure 1
plot(negH, -log(1-negH)); % plot de cost en fonction de h
hold on 
title('Cost Fuction (y = 0)'); % titre du plot 1
xlabel('H(x)'); % label des X
ylabel('-log(1-h)'); % label des Y

figure('name','cost function 2'); % titre de la figure 2
plot(posH, -log(posH)); % plot de cost en fonction de h
hold on 
title('Cost Fuction (y = 1)'); % titre du plot 2
xlabel('H(x)'); % label des X
ylabel('-log(h)'); % label des Y