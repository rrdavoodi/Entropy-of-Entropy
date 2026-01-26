function EEG_Phase_Entropy_Analysis()
clc; close all; warning off;
% initial setting

channelNames = {'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8'};
num_channels = length(channelNames);

% 2. data loading
[input_file, input_path] = uigetfile('*.mat', 'Select EEG Data File');
if isequal(input_file, 0)
    error('Operation canceled by user');
end

data = load(fullfile(input_path, input_file));
myCell = data.C;

% 3. data structure evaluation
[num_emotions, num_subjects] = size(myCell);

if size(myCell{1,1},2) == num_channels && size(myCell{1,1},1) > num_channels
    for emo = 1:num_emotions
        for sub = 1:num_subjects
            myCell{emo,sub} = myCell{emo,sub}';
        end
    end
end

% 4. processing parameters
fs = 128;
total_duration = 300;
analysis_duration = 120;
total_samples = total_duration * fs;
analysis_samples = analysis_duration * fs;

if size(myCell{1,1},2) < total_samples
    analysis_samples = min(analysis_samples, size(myCell{1,1},2));
    start_sample = 1;
else
    start_sample = floor((total_samples - analysis_samples)/2) + 1;
end
end_sample = start_sample + analysis_samples - 1;

% 5. shannon entropy
function se = shannon_entropy(signal, bins)
    signal = (signal - min(signal)) / (max(signal) - min(signal) + eps);
    [counts, ~] = histcounts(signal, bins);
    prob = counts(counts > 0) / sum(counts);
    se = -sum(prob .* log2(prob + eps));
end

% 6.phase extraction and claculation of phase entropy
results = struct();
results.phase_entropy = cell(num_emotions, num_subjects);

for emo = 1:num_emotions
    for sub = 1:num_subjects
        for ch = 1:num_channels
            try
                signal = myCell{emo,sub}(ch, start_sample:end_sample);
                analytic_signal = hilbert(signal);
                phase = angle(analytic_signal);
                results.phase_entropy{emo,sub}(ch) = shannon_entropy(phase, 10);
            catch
                results.phase_entropy{emo,sub}(ch) = NaN;
            end
        end
    end
end

% 7. statistical test
stats = struct();
mean_entropy = zeros(num_emotions, num_channels);
normality = zeros(num_emotions, num_channels);
alpha = 0.05;

for emo = 1:num_emotions
    for ch = 1:num_channels
        values = cellfun(@(x) x(ch), results.phase_entropy(emo,:), 'UniformOutput', false);
        values = cell2mat(values(~cellfun(@isempty, values)));
        values = values(~isnan(values));
        if ~isempty(values)
            mean_entropy(emo,ch) = mean(values);
            [~, p] = kstest(zscore(values));
            normality(emo,ch) = p > alpha;
        else
            mean_entropy(emo,ch) = NaN;
        end
    end
end

for ch = 1:num_channels
    all_data = [];
    groups = [];
    for emo = 1:num_emotions
        values = cellfun(@(x) x(ch), results.phase_entropy(emo,:), 'UniformOutput', false);
        values = cell2mat(values(~cellfun(@isempty, values)));
        values = values(~isnan(values));
        all_data = [all_data; values(:)];
        groups = [groups; emo * ones(length(values), 1)];
    end

    % removal of NaN
    valid_idx = ~isnan(all_data);
    all_data = all_data(valid_idx);
    groups = groups(valid_idx);

    if sum(normality(:,ch)) == num_emotions
        [p, tbl, stat_mat] = anova1(all_data, groups, 'off');
        stats(ch).test = 'ANOVA';
        stats(ch).p_value = p;
        stats(ch).df = tbl{2,3};
        stats(ch).f_value = tbl{2,5};
        [c,~,~,gnames] = multcompare(stat_mat, 'Display','off');
        stats(ch).posthoc = array2table(c, 'VariableNames', ...
            {'Group1','Group2','LowerCI','Estimate','UpperCI','PValue'});
        stats(ch).posthoc.Group1 = gnames(stats(ch).posthoc.Group1);
        stats(ch).posthoc.Group2 = gnames(stats(ch).posthoc.Group2);
    else
        [p, tbl, ~] = kruskalwallis(all_data, groups, 'off');
        stats(ch).test = 'Kruskal-Wallis';
        stats(ch).p_value = p;
        stats(ch).df = tbl{2,3};
        stats(ch).chi2 = tbl{2,5};
        [pvals, padj] = dunntest(all_data, groups);
        stats(ch).posthoc = array2table(padj, 'VariableNames', ...
            arrayfun(@(x) sprintf('Emotion_%d',x), 1:num_emotions, 'UniformOutput', false));
    end
end

% 8. saving results
[output_file, output_path] = uiputfile('*.xlsx', 'Save Results As', 'Phase_Entropy_Results.xlsx');
if isequal(output_file, 0)
    error('Operation canceled by user');
end

writer = @(data, sheet) writetable(data, fullfile(output_path, output_file), ...
    'Sheet', sheet, 'WriteMode', 'overwritesheet');

mean_table = array2table(mean_entropy', 'RowNames', channelNames, ...
    'VariableNames', arrayfun(@(x) sprintf('Emotion_%d',x), 1:num_emotions, 'UniformOutput', false));
writer(mean_table, 'Mean_Phase_Entropy');

normality_table = array2table(normality', 'RowNames', channelNames, ...
    'VariableNames', arrayfun(@(x) sprintf('Emotion_%d',x), 1:num_emotions, 'UniformOutput', false));
writer(normality_table, 'Normality');

for ch = 1:num_channels
    stats_table = table();
    stats_table.Test = {stats(ch).test};
    stats_table.pValue = stats(ch).p_value;
    if strcmp(stats(ch).test, 'ANOVA')
        stats_table.DF = stats(ch).df;
        stats_table.F = stats(ch).f_value;
    else
        stats_table.DF = stats(ch).df;
        stats_table.Chi2 = stats(ch).chi2;
    end
    writer(stats_table, ['Stats_' channelNames{ch}]);
    writer(stats(ch).posthoc, ['Posthoc_' channelNames{ch}]);
end

% 9. displaying the results
figure('Name','Phase Entropy Analysis','NumberTitle','off','Position',[100 100 1200 800]);

figure;
bar(mean_entropy');
set(gca,'XTick',1:num_channels,'XTickLabel',channelNames);
xlabel('Channels');
ylabel('Mean Phase Entropy');
title('Mean Phase Entropy Across Channels');
legend(arrayfun(@(x) sprintf('Emotion %d',x),1:num_emotions,'UniformOutput',false));

figure;
plot_samples = min(fs*2, size(myCell{1,1},2));
plot(myCell{1,1}(1, 1:plot_samples));
xlabel('Samples');
ylabel('Amplitude');
title(['Sample Signal - ' channelNames{1}]);
grid on;

%  p-value
figure;
p_values = arrayfun(@(x) x.p_value, stats);
bar(p_values);
hold on;
plot([0 num_channels+1], [0.05 0.05], 'r--');
set(gca,'XTick',1:num_channels,'XTickLabel',channelNames);
xlabel('Channels');
ylabel('p-value');
title('Statistical Test Results');
legend('p-values', 'Significance Level (0.05)');
ylim([0 1]);

% DUNN test
function [pvals, padj] = dunntest(data, groups)
    groups = categorical(groups);
    group_list = unique(groups);
    k = length(group_list);
    pvals = zeros(k);
    for i = 1:k
        for j = i+1:k
            [pvals(i,j), ~] = ranksum(data(groups==group_list(i)), data(groups==group_list(j)));
        end
    end
    padj = pvals * (k*(k-1)/2); % Bonferroni correction
end

disp('=== Phase Entropy Analysis Completed ===');
end
