% Schritt 1: Starte Python-Benchmark
py.importlib.import_module('benchmark');
py.benchmark.run_benchmark();

% Schritt 2: Lade Python-Ergebnisse
data = readtable('benchmark_results.csv');

% Schritt 3: MATLAB-Benchmark mit denselben Größen
disp("Starte MATLAB-Benchmark...");
ns = data.n;
num_runs = 20;
matlab_mean = zeros(size(ns));
matlab_std = zeros(size(ns));
matlab_ci = zeros(size(ns));

for i = 1:length(ns)
    n = ns(i);
    times = zeros(num_runs, 1);
    for r = 1:num_runs
        A = rand(n);
        B = rand(n);
        t1 = tic;
        C = A * B;
        times(r) = toc(t1);
    end
    matlab_mean(i) = mean(times);
    matlab_std(i) = std(times, 1);
    ci = tinv(0.975, num_runs-1) * matlab_std(i) / sqrt(num_runs);
    matlab_ci(i) = ci;
end

% Ergebnisse hinzufügen
data.matlab_mean = matlab_mean;
data.matlab_std = matlab_std;
data.matlab_ci = matlab_ci;

% Als neue CSV speichern (optional)
writetable(data, 'benchmark_results_all.csv');

% data = readtable("benchmark_results_all.csv");
%% Plot 1: Laufzeit
figure;
loglog(data.n, data.numpy_mean, '-o', 'DisplayName', 'NumPy'); hold on;
loglog(data.n, data.torch_mean, '-x', 'DisplayName', 'PyTorch');
loglog(data.n, data.tf_mean, '-s', 'DisplayName', 'TensorFlow');
loglog(data.n, data.matlab_mean, '-d', 'DisplayName', 'MATLAB');
xlabel('Matrixgröße n');
ylabel('Mittlere Zeit (s)');
title('Matrix-Multiplikation – Laufzeit über 20 Wiederholungen');
legend('Location', 'northwest');
grid on;

% === Schritt 4: Plot 2 – GFLOP/s mit 95 % Konfidenzintervall ===
flops = 2 * (data.n .^ 3);  % FLOPs pro Multiplikation
libs = {'numpy', 'torch', 'tf', 'matlab'};

figure; hold on;
for i = 1:length(libs)
    lib = libs{i};
    t_mean = data.([lib '_mean']);
    t_ci = data.([lib '_ci']);
    speed = flops ./ t_mean;
    ci_rel = t_ci ./ t_mean;
    errorbar(data.n, speed / 1e9, (speed .* ci_rel) / 1e9, ...
        'DisplayName', lib, 'LineWidth', 1.5);
end
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Matrixgröße n');
ylabel('GFLOP/s');
title('Rechenleistung mit 95% Konfidenzintervall');
legend('Location', 'northwest');
grid on;

% === Schritt 5: Durchschnitt über die größten 4 Größen ===
idx = height(data) - 3 : height(data);
fprintf("\nDurchschnittliche GFLOP/s über größte 4 Matrixgrößen:\n");
for i = 1:length(libs)
    lib = libs{i};
    t_mean_last4 = data.([lib '_mean'])(idx);
    avg_speed = mean(flops(idx) ./ t_mean_last4);
    fprintf('%-8s: %.2f GFLOP/s\n', lib, avg_speed / 1e9);
end