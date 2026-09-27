%% analyze_cross_split_leakage_31800.m
% Detailed offline analysis of highly similar cross-split pairs.
% Hardware used: NO
% Training performed: NO
% Original Dataset modified: NO

clear;
clc;
close all;

%% Paths
projectFolder = pwd;
resultsFolder = fullfile(projectFolder, ...
    'results_dataset_31800_temporal_inspection');

pairsFile = fullfile(resultsFolder, ...
    'cross_split_high_similarity_pairs.csv');

assert(isfile(pairsFile), ...
    'Cross-split pair file was not found: %s', pairsFile);

fprintf('\n');
fprintf('============================================================\n');
fprintf('CROSS-SPLIT LEAKAGE DETAIL - SDR DATASET 31800\n');
fprintf('============================================================\n');
fprintf('Mode: OFFLINE DATASET INSPECTION\n');
fprintf('Hardware used: NO\n');
fprintf('Training performed: NO\n');
fprintf('Original Dataset modified: NO\n');
fprintf('\nInput pairs file:\n%s\n', pairsFile);
fprintf('============================================================\n');

%% Load high-similarity pairs
P = readtable(pairsFile, 'TextType', 'string');

requiredColumns = {
    'Comparison'
    'QueryDatasetRow'
    'ReferenceDatasetRow'
    'QueryCaptureOrder'
    'ReferenceCaptureOrder'
    'QuerySplit'
    'ReferenceSplit'
    'QueryTx'
    'ReferenceTx'
    'QueryModulation'
    'ReferenceModulation'
    'QueryClass'
    'ReferenceClass'
    'Similarity'
};

for columnIndex = 1:numel(requiredColumns)
    assert(ismember(requiredColumns{columnIndex}, ...
        P.Properties.VariableNames), ...
        'Required column is missing: %s', ...
        requiredColumns{columnIndex});
end

P.Similarity = double(P.Similarity);
P.SameModulation = P.QueryModulation == P.ReferenceModulation;
P.SameTx = P.QueryTx == P.ReferenceTx;
P.SameClass = P.QueryClass == P.ReferenceClass;
P.SameModulationAndTx = P.SameModulation & P.SameTx;
P.Review0980 = P.Similarity >= 0.980;
P.Severe0995 = P.Similarity >= 0.995;
P.NearExact0999 = P.Similarity >= 0.999;
P.CaptureOrderDistance = abs(double(P.QueryCaptureOrder) ...
    - double(P.ReferenceCaptureOrder));

fprintf('\nPAIR SUMMARY\n');
fprintf('============\n');
fprintf('Pairs >= 0.980: %d\n', height(P));
fprintf('Pairs >= 0.995: %d\n', sum(P.Severe0995));
fprintf('Pairs >= 0.999: %d\n', sum(P.NearExact0999));
fprintf('Same modulation: %d\n', sum(P.SameModulation));
fprintf('Same TX: %d\n', sum(P.SameTx));
fprintf('Same class: %d\n', sum(P.SameClass));
fprintf('Same modulation and TX: %d\n', ...
    sum(P.SameModulationAndTx));
fprintf('Different class pairs: %d\n', sum(~P.SameClass));

%% Compatible aggregation helper data: query modulation
[modGroup, modNames] = findgroups(P.QueryModulation);
modTotal = splitapply(@numel, P.Similarity, modGroup);
modSevere = splitapply(@sum, double(P.Severe0995), modGroup);
modNearExact = splitapply(@sum, double(P.NearExact0999), modGroup);
modSameMod = splitapply(@sum, double(P.SameModulation), modGroup);
modSameTx = splitapply(@sum, double(P.SameTx), modGroup);
modMeanSimilarity = splitapply(@mean, P.Similarity, modGroup);
modMaximumSimilarity = splitapply(@max, P.Similarity, modGroup);
modMedianOrderDistance = splitapply(@median, ...
    P.CaptureOrderDistance, modGroup);

leakageByModulation = table(modNames, modTotal, modSevere, ...
    modNearExact, modSameMod, modSameTx, modMeanSimilarity, ...
    modMaximumSimilarity, modMedianOrderDistance, ...
    'VariableNames', {
        'ModulationName'
        'PairsAbove0980'
        'PairsAbove0995'
        'PairsAbove0999'
        'SameModulationPairs'
        'SameTxPairs'
        'MeanSimilarity'
        'MaximumSimilarity'
        'MedianCaptureOrderDistance'
    });

leakageByModulation = sortrows(leakageByModulation, ...
    {'PairsAbove0999','PairsAbove0995'}, {'descend','descend'});

fprintf('\nLEAKAGE BY QUERY MODULATION\n');
fprintf('===========================\n');
disp(leakageByModulation);

%% Aggregation by comparison
[comparisonGroup, comparisonNames] = findgroups(P.Comparison);
comparisonTotal = splitapply(@numel, P.Similarity, comparisonGroup);
comparisonSevere = splitapply(@sum, ...
    double(P.Severe0995), comparisonGroup);
comparisonNearExact = splitapply(@sum, ...
    double(P.NearExact0999), comparisonGroup);
comparisonSameMod = splitapply(@sum, ...
    double(P.SameModulation), comparisonGroup);
comparisonSameTx = splitapply(@sum, ...
    double(P.SameTx), comparisonGroup);
comparisonMean = splitapply(@mean, P.Similarity, comparisonGroup);
comparisonMaximum = splitapply(@max, P.Similarity, comparisonGroup);

leakageByComparison = table(comparisonNames, comparisonTotal, ...
    comparisonSevere, comparisonNearExact, comparisonSameMod, ...
    comparisonSameTx, comparisonMean, comparisonMaximum, ...
    'VariableNames', {
        'Comparison'
        'PairsAbove0980'
        'PairsAbove0995'
        'PairsAbove0999'
        'SameModulationPairs'
        'SameTxPairs'
        'MeanSimilarity'
        'MaximumSimilarity'
    });

fprintf('\nLEAKAGE BY SPLIT COMPARISON\n');
fprintf('===========================\n');
disp(leakageByComparison);

%% Aggregation by query TX
[txGroup, txNames] = findgroups(P.QueryTx);
txTotal = splitapply(@numel, P.Similarity, txGroup);
txSevere = splitapply(@sum, double(P.Severe0995), txGroup);
txNearExact = splitapply(@sum, double(P.NearExact0999), txGroup);
txSameTx = splitapply(@sum, double(P.SameTx), txGroup);
txMean = splitapply(@mean, P.Similarity, txGroup);
txMaximum = splitapply(@max, P.Similarity, txGroup);

leakageByTx = table(txNames, txTotal, txSevere, txNearExact, ...
    txSameTx, txMean, txMaximum, ...
    'VariableNames', {
        'QueryTx'
        'PairsAbove0980'
        'PairsAbove0995'
        'PairsAbove0999'
        'SameTxPairs'
        'MeanSimilarity'
        'MaximumSimilarity'
    });

fprintf('\nLEAKAGE BY QUERY TX\n');
fprintf('===================\n');
disp(leakageByTx);

%% Aggregation by query class
[classGroup, classNames] = findgroups(P.QueryClass);
classTotal = splitapply(@numel, P.Similarity, classGroup);
classSevere = splitapply(@sum, double(P.Severe0995), classGroup);
classNearExact = splitapply(@sum, ...
    double(P.NearExact0999), classGroup);
classDifferent = splitapply(@sum, double(~P.SameClass), classGroup);
classMean = splitapply(@mean, P.Similarity, classGroup);
classMaximum = splitapply(@max, P.Similarity, classGroup);

leakageByClass = table(classNames, classTotal, classSevere, ...
    classNearExact, classDifferent, classMean, classMaximum, ...
    'VariableNames', {
        'QueryClass'
        'PairsAbove0980'
        'PairsAbove0995'
        'PairsAbove0999'
        'DifferentClassPairs'
        'MeanSimilarity'
        'MaximumSimilarity'
    });

fprintf('\nLEAKAGE BY QUERY CLASS\n');
fprintf('======================\n');
disp(leakageByClass);

%% Near-exact pairs
nearExactPairsTable = P(P.NearExact0999,:);
nearExactPairsTable = sortrows(nearExactPairsTable, ...
    'Similarity','descend');

fprintf('\nTOP NEAR-EXACT CROSS-SPLIT PAIRS\n');
fprintf('================================\n');
if isempty(nearExactPairsTable)
    fprintf('No pairs reached similarity 0.999.\n');
else
    disp(nearExactPairsTable(1:min(30,height(nearExactPairsTable)),:));
end

%% Unique rows affected
allReviewRows = unique([double(P.QueryDatasetRow); ...
    double(P.ReferenceDatasetRow)]);
severeRows = unique([double(P.QueryDatasetRow(P.Severe0995)); ...
    double(P.ReferenceDatasetRow(P.Severe0995))]);
nearExactRows = unique([double(P.QueryDatasetRow(P.NearExact0999)); ...
    double(P.ReferenceDatasetRow(P.NearExact0999))]);

rowImpactSummary = table(height(P), sum(P.Severe0995), ...
    sum(P.NearExact0999), numel(allReviewRows), numel(severeRows), ...
    numel(nearExactRows), ...
    'VariableNames', {
        'PairCountAbove0980'
        'PairCountAbove0995'
        'PairCountAbove0999'
        'UniqueRowsAbove0980'
        'UniqueRowsAbove0995'
        'UniqueRowsAbove0999'
    });

fprintf('\nUNIQUE DATASET ROW IMPACT\n');
fprintf('=========================\n');
disp(rowImpactSummary);

%% Decision indicators
nearExactSameModulationPct = NaN;
nearExactSameTxPct = NaN;
nearExactDifferentClassCount = 0;

if ~isempty(nearExactPairsTable)
    nearExactSameModulationPct = 100 * ...
        mean(nearExactPairsTable.SameModulation);
    nearExactSameTxPct = 100 * mean(nearExactPairsTable.SameTx);
    nearExactDifferentClassCount = sum(~nearExactPairsTable.SameClass);
end

decisionSummary = table(height(P), sum(P.Severe0995), ...
    sum(P.NearExact0999), nearExactSameModulationPct, ...
    nearExactSameTxPct, nearExactDifferentClassCount, ...
    'VariableNames', {
        'PairsAbove0980'
        'PairsAbove0995'
        'PairsAbove0999'
        'NearExactSameModulationPct'
        'NearExactSameTxPct'
        'NearExactDifferentClassCount'
    });

fprintf('\nDECISION SUMMARY\n');
fprintf('================\n');
disp(decisionSummary);

%% Save outputs
writetable(P, fullfile(resultsFolder, ...
    'cross_split_pairs_enriched.csv'));
writetable(leakageByModulation, fullfile(resultsFolder, ...
    'leakage_by_modulation.csv'));
writetable(leakageByComparison, fullfile(resultsFolder, ...
    'leakage_by_comparison.csv'));
writetable(leakageByTx, fullfile(resultsFolder, ...
    'leakage_by_tx.csv'));
writetable(leakageByClass, fullfile(resultsFolder, ...
    'leakage_by_class.csv'));
writetable(nearExactPairsTable, fullfile(resultsFolder, ...
    'near_exact_cross_split_pairs.csv'));
writetable(rowImpactSummary, fullfile(resultsFolder, ...
    'cross_split_row_impact_summary.csv'));
writetable(decisionSummary, fullfile(resultsFolder, ...
    'cross_split_decision_summary.csv'));

save(fullfile(resultsFolder, ...
    'cross_split_leakage_detail.mat'), ...
    'P','leakageByModulation','leakageByComparison', ...
    'leakageByTx','leakageByClass','nearExactPairsTable', ...
    'rowImpactSummary','decisionSummary','allReviewRows', ...
    'severeRows','nearExactRows','-v7.3');

%% Charts
figure('Color','white','Name','Leakage by Modulation', ...
    'NumberTitle','off','Position',[100 100 1300 650]);
bar(categorical(leakageByModulation.ModulationName), ...
    [leakageByModulation.PairsAbove0980, ...
     leakageByModulation.PairsAbove0995, ...
     leakageByModulation.PairsAbove0999]);
grid on;
xlabel('Query Modulation');
ylabel('Cross-Split Pair Count');
title('Cross-Split Similarity by Modulation');
legend({'Similarity >= 0.980','Similarity >= 0.995', ...
    'Similarity >= 0.999'},'Location','best');
xtickangle(45);
exportgraphics(gcf,fullfile(resultsFolder, ...
    'leakage_by_modulation.png'),'Resolution',200);

figure('Color','white','Name','Leakage by Comparison', ...
    'NumberTitle','off','Position',[100 100 1000 600]);
bar(categorical(leakageByComparison.Comparison), ...
    [leakageByComparison.PairsAbove0980, ...
     leakageByComparison.PairsAbove0995, ...
     leakageByComparison.PairsAbove0999]);
grid on;
xlabel('Split Comparison');
ylabel('Pair Count');
title('Cross-Split Similarity by Split Comparison');
legend({'Similarity >= 0.980','Similarity >= 0.995', ...
    'Similarity >= 0.999'},'Location','best');
xtickangle(15);
exportgraphics(gcf,fullfile(resultsFolder, ...
    'leakage_by_comparison.png'),'Resolution',200);

%% Final status
fprintf('\n============================================================\n');
fprintf('CROSS-SPLIT LEAKAGE DETAIL COMPLETED\n');
fprintf('============================================================\n');
fprintf('Pairs >= 0.980: %d\n', height(P));
fprintf('Pairs >= 0.995: %d\n', sum(P.Severe0995));
fprintf('Pairs >= 0.999: %d\n', sum(P.NearExact0999));
fprintf('Unique affected rows >= 0.999: %d\n', ...
    numel(nearExactRows));
fprintf('Near-exact different-class pairs: %d\n', ...
    nearExactDifferentClassCount);
fprintf('\nResults folder:\n%s\n', resultsFolder);
fprintf('============================================================\n');
