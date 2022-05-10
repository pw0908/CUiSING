clear; clc;
set(groot, ...
    'DefaultAxesTickLabelInterpreter', 'LaTeX', ...
    'DefaultLegendInterpreter', 'LaTeX', ...
    'DefaultLineLineWidth', 1.5, ...
    'DefaultTextInterpreter', 'LaTeX', ...
    'DefaultTextHorizontalAlignment', 'center')
close all

% Set x and y dim (n), and number of spins n^2
n = 100;  N = n^2;

% J is the interaction strength
% Sample freq sets how frequently to print the visualization
J = 0.5; n_iters = 10000; sample_freq = 1;

% Initialize the vectors for storing magnetization and energy
ms = zeros(n_iters + 1, 1); Es = ms;

% Initialize matrix of spins (randomly assigned +-1)
A = 1 - 2 * randi([0, 1], n);

% Store spin values as row, col, val
[row, col, val] = find(A); pos = [row, col, val];

% Calculate initial m and E
ms(1) = sum(A, 'all') / N;
Es(1) = -J * pos(:, 3)' * diff_neigh(pos, A, n)/2;

% Set up colormap for outputting system visual
cmap = [1, 0.2, 0.2; 1, 1, 1; 0, 1, 0.5]; 
figure; colormap(cmap);
imagesc(A); axis square
print('initial.png','-dpng',['-r' '350']);
hold off;
figure; colormap(cmap);

% Outer loop for MC iterations
tic;
for i = 2:n_iters + 1
    % In each MC iteration, attempt to flip all spins, using metropolis
    for j = 1:N
        dE = 2 * J * pos(j, 3) * sum(...
            [A(mod(pos(j, 1) - 2, n) + 1, pos(j, 2)), ...
            A(mod(pos(j, 1), n) + 1, pos(j, 2)), ...
            A(pos(j, 1), mod(pos(j, 2) - 2, n) + 1), ...
            A(pos(j, 1), mod(pos(j, 2), n) + 1)]);
        if dE <= 0 || rand() <= exp(-dE)
            A(pos(j, 1), pos(j, 2)) = -pos(j, 3);
        end
    end
    
    % Update the row, col, val since some were flipped
    [row, col, val] = find(A); pos = [row, col, val];
    
    % Calculate observables
    ms(i) = sum(A, 'all') / N;
    Es(i) = J * pos(:, 3)' * diff_neigh(pos, A, n)/2;
    
    % Output visual if multiple of sample_freq
%     if mod(i - 1, sample_freq) == 0
%         imagesc(A); axis square; title(i); figure(gcf);
%     end
end
toc;
print('final.png','-dpng',['-r' '350']);

figure; plot(ms, 'k.', 'MarkerSize', 10); axis square
xlabel('Iteration'); ylabel('$m$'); xlim([0, n_iters]); %ylim([-0.25, 0.25])
print('m_vs_i.png','-dpng',['-r' '350']);
figure; plot(Es, 'k.', 'MarkerSize', 10); axis square
xlabel('Iteration'); ylabel('$E$'); xlim([0, n_iters])
print('E_vs_i.png','-dpng',['-r' '350']);

fprintf('Average m = %.3f\n',mean(ms(600:end)));
fprintf('Average E = %.3f\n',mean(Es(600:end)));

% Function for calculating the difference between up and down neighbors
% of a particular spin. This is useful for calculating the energy change
% when flipping a spin, since it only depends on the surrounding spins
function dnn = diff_neigh(xy, A, n)
dnn = -sum([A(mod(xy(:, 1) - 2, n) + 1 + n * (xy(:, 2) - 1)), ...
    A(mod(xy(:, 1), n) + 1 + n * (xy(:, 2) - 1)), ...
    A(xy(:, 1) + n * mod(xy(:, 2) - 2, n)), ...
    A(xy(:, 1) + n * mod(xy(:, 2), n))], 2);
end