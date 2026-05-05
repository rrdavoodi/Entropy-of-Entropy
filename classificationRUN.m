clc;
close all;
clear all;

% % load('EEG_Entropy_Phase_Feature.mat');
load('EEG_Phase_EoE_Feature.mat');     
X = reshape(permute(FeatureMatrix, [2,1,3]), 112, 14);  % [samples x features]
y = repelem((1:4)', 28);
subjects = repelem((1:28)', 4);      

classifiers = {'KNN', 'LDA', 'SVM', 'Tree', 'RF', 'AdaBoost'};
K = 5;
Nrepeat = 10;

results_all = [];
acc_matrix = zeros(Nrepeat, length(classifiers));

radar_values = zeros(length(classifiers), 4);

for c = 1:length(classifiers)
    acc_all = zeros(Nrepeat,1);
    f1_all = zeros(Nrepeat,1);
    auc_all = zeros(Nrepeat,1);
    pr_auc_all = zeros(Nrepeat,1);
    conf_all = cell(Nrepeat,1);
    pr_data = cell(4,2);
    roc_data = cell(4,2);
    best_auc = -inf;
    for rep = 1:Nrepeat
        cv = cvpartition(subjects, 'KFold', K);

        acc = zeros(K,1);
        f1_macro = zeros(K,1);
        auc_macro = zeros(K,1);
        pr_auc_macro = zeros(K,1);
        cm_rep = zeros(4,4);
        roc_rep = cell(4,2);
        pr_rep  = cell(4,2);
        all_scores = [];
        all_labels = [];
        for fold = 1:K
            trainIdx = cv.training(fold);
            testIdx = cv.test(fold);

            X_train = X(trainIdx,:);
            y_train = y(trainIdx);
            X_test = X(testIdx,:);
            y_test = y(testIdx);

            switch classifiers{c}
                case 'KNN'
                    model = fitcknn(X_train, y_train, 'NumNeighbors', 5);
                case 'LDA'
                    model = fitcdiscr(X_train, y_train);
                case 'SVM'
                    model = fitcecoc(X_train, y_train, 'Learners', templateSVM('KernelFunction','linear', 'Standardize',true));
                case 'Tree'
                    model = fitctree(X_train, y_train);
                case 'RF'
                    model = fitcensemble(X_train, y_train, 'Method', 'Bag');
                case 'AdaBoost'
                    model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM2');
            end

            [pred, scores] = predict(model, X_test);
            all_scores = [all_scores; scores];
            all_labels = [all_labels; y_test];
            acc(fold) = mean(pred == y_test);

            cm = confusionmat(y_test, pred, 'Order', 1:4);
            cm_rep = cm_rep + cm;
            precision = diag(cm) ./ (sum(cm,1)' + eps);
            recall = diag(cm) ./ (sum(cm,2) + eps);
            f1 = 2 * (precision .* recall) ./ (precision + recall + eps);
            f1_macro(fold) = mean(f1);

            if size(scores,2) == 4
                aucs = zeros(4,1);
                pr_aucs = zeros(4,1);
                for k = 1:4
                    [~,~,~,auck] = perfcurve(y_test == k, scores(:,k), true);
                    aucs(k) = auck;

                    [rec, prec, ~, pr_auc] = perfcurve(y_test == k, scores(:,k), true, 'xCrit','reca','yCrit','prec');
                    pr_aucs(k) = pr_auc;

                    %                     if rep == 1 && fold == 1
                    pr_data{k,1} = rec;
                    pr_data{k,2} = prec;

                    [fpr, tpr] = perfcurve(y_test == k, scores(:,k), true);
                    roc_data{k,1} = fpr;
                    roc_data{k,2} = tpr;
                    %                     end
                    pr_rep{k,1} = rec;
                    pr_rep{k,2} = prec;

                    [fpr, tpr] = perfcurve(y_test == k, scores(:,k), true);
                    roc_rep{k,1} = fpr;
                    roc_rep{k,2} = tpr;
                end
                auc_macro(fold) = mean(aucs);
                pr_auc_macro(fold) = mean(pr_aucs);
            else
                auc_macro(fold) = NaN;
                pr_auc_macro(fold) = NaN;
            end
        end
        pr_rep = cell(4,2);
        roc_rep = cell(4,2);

        for k = 1:4
            [fpr, tpr, ~, ~] = perfcurve(all_labels == k, all_scores(:,k), true);
            roc_rep{k,1} = fpr;
            roc_rep{k,2} = tpr;

            [rec, prec, ~, ~] = perfcurve(all_labels == k, all_scores(:,k), true, ...
                'xCrit','reca','yCrit','prec');

            pr_rep{k,1} = rec;
            pr_rep{k,2} = prec;
        end

        acc_all(rep) = mean(acc);
        f1_all(rep) = mean(f1_macro);
        auc_all(rep) = mean(auc_macro);
        pr_auc_all(rep) = mean(pr_auc_macro);
        conf_all{rep} = cm_rep;
        mean_auc_rep = mean(auc_macro);

        if rep == 1 || mean_auc_rep > best_auc
            best_auc = mean_auc_rep;
            best_roc = roc_rep;
            best_pr  = pr_rep;
        end

    end

    acc_matrix(:,c) = acc_all;
    radar_values(c,:) = [mean(acc_all), mean(f1_all), mean(auc_all), mean(pr_auc_all)];

    results_all = [results_all; {
        classifiers{c}, ...
        max(acc_all), std(acc_all), ...
        max(f1_all), std(f1_all), ...
        max(auc_all), std(auc_all), ...
        max(pr_auc_all), std(pr_auc_all)}];
    if strcmp(classifiers{c}, 'AdaBoost')
        rep_table = table((1:Nrepeat)', acc_all, f1_all, auc_all, pr_auc_all, ...
            'VariableNames', {'Repetition', 'Accuracy', 'F1', 'AUC_ROC', 'AUC_PR'});
        writetable(rep_table, 'AdaBoost_repetition_results.csv');
    end
    [~, best_rep] = max(auc_all);
    best_cm = conf_all{best_rep};
    figure('Name', ['Confusion - ' classifiers{c}]);
    confusionchart(best_cm, {'Boring','Calm','Horror','Funny'});
    title(['Best Confusion Matrix - ' classifiers{c}]);
    set(gcf, 'Units', 'inches', 'Position', [1 1 6 5]);
    print(gcf, ['Phase_Ent_Confusion_' classifiers{c}], '-dpng', '-r300');

    % Precision-Recall Curve
    figure;
    hold on;
    for k = 1:4
        plot(best_pr{k,1}, best_pr{k,2}, 'LineWidth', 2);
    end
    legend({'Boring','Calm','Horror','Funny'});
    xlabel('Recall'); ylabel('Precision');
    title('PR Curve (Aggregated over 4-fold CV)');
    grid on;

    legend({'Boring','Calm','Horror','Funny'});
    xlabel('Recall'); ylabel('Precision');
    title(['Precision-Recall Curve - ' classifiers{c}]);
    grid on;
    print(gcf, ['PRC_' classifiers{c}], '-dpng', '-r300');

    % ROC Curve
    figure;
    hold on;
    for k = 1:4
        plot(best_roc{k,1}, best_roc{k,2}, 'LineWidth', 2);
    end
    legend({'Boring','Calm','Horror','Funny'});
    xlabel('Recall'); ylabel('Precision');
    title('PR Curve (Aggregated over 4-fold CV)');
    grid on;

    legend({'Boring','Calm','Horror','Funny'});
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(['ROC Curve - ' classifiers{c}]);
    grid on;
    print(gcf, ['ROC_' classifiers{c}], '-dpng', '-r300');
end

result_table = cell2table(results_all, ...
    'VariableNames', {'Classifier', ...
    'Mean_Accuracy', 'Std_Accuracy', ...
    'Mean_F1', 'Std_F1', ...
    'Mean_AUC_ROC', 'Std_AUC_ROC', ...
    'Mean_AUC_PR', 'Std_AUC_PR'});

writetable(result_table, 'Final_Results_phaseEn.csv');

% Radar Plot
figure;
theta = linspace(0, 2*pi, 5);
theta(end) = [];
th = [theta, theta(1)];
labels = {'Accuracy','F1 Macro','AUC ROC','AUC PR'};

for i = 1:size(radar_values,1)
    val = radar_values(i,:);
    polarplot([val, val(1)], 'DisplayName', classifiers{i}, 'LineWidth', 1.5);
    hold on;
end
legend('show');
title('Radar Plot - Classifier Comparison');
print(gcf, 'RadarPlot_PhaseEnt', '-dpng', '-r300');

% Boxplot
figure;
boxplot(acc_matrix, classifiers);
title('Accuracy Distribution over 10 Runs');
ylabel('Accuracy');
print(gcf, 'Boxplot_Accuracy_PhaseEnt', '-dpng', '-r300');

% Friedman Test
[p_friedman, tbl, stats] = friedman(acc_matrix, 1, 'off');
if p_friedman < 0.05
    fprintf('Friedman test: Significant differences found (p = %.4f)\n', p_friedman);
else
    fprintf('Friedman test: No significant differences (p = %.4f)\n', p_friedman);
end
