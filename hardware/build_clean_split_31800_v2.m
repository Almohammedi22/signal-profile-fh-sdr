%% build_clean_split_31800_v2.m
% Build a leakage-controlled split for the 31,800 real SDR captures.
%
% TYPE: OFFLINE DATASET PREPARATION
% Hardware used: NO
% Neural-network training: NO
% Original Dataset modified: NO
%
% Method:
%   1. Read the previously detected cross-split pairs.
%   2. Keep pairs with similarity >= 0.995.
%   3. Join connected captures into leakage groups.
%   4. Assign each whole group to exactly one split.
%   5. Preserve all 31,800 captures.
%   6. Target, per Modulation x TX:
%          Train = 350, Validation = 90, Test = 90.
%   7. Save a new MAT file with originalSplit and cleanSplit.
%
% The original real_sdr_dataset_31800.mat is never overwritten.

clear;
clc;
close all;

rng(42,'twister');

%% =========================================================
% Paths and settings
% ==========================================================

projectFolder = pwd;

datasetFile = fullfile( ...
    projectFolder, ...
    'results_real_dataset_31800', ...
    'dataset_31800_current', ...
    'real_sdr_dataset_31800.mat');

leakageFile = fullfile( ...
    projectFolder, ...
    'results_dataset_31800_temporal_inspection', ...
    'cross_split_pairs_enriched.csv');

if ~isfile(leakageFile)
    leakageFile = fullfile( ...
        projectFolder, ...
        'results_dataset_31800_temporal_inspection', ...
        'cross_split_high_similarity_pairs.csv');
end

resultsFolder = fullfile( ...
    projectFolder, ...
    'results_dataset_31800_clean_split_v2');

outputDatasetFile = fullfile( ...
    resultsFolder, ...
    'real_sdr_dataset_31800_clean_split.mat');

similarityThreshold = 0.995;
targetPerCombination = [350 90 90];
splitLabels = ["train" "validation" "test"];

assert(isfile(datasetFile),'Dataset file was not found.');
assert(isfile(leakageFile),'Leakage-pair file was not found.');

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

fprintf('\n');
fprintf('============================================================\n');
fprintf('BUILD CLEAN SPLIT V2 - REAL SDR DATASET 31800\n');
fprintf('============================================================\n');
fprintf('Mode: OFFLINE DATASET PREPARATION\n');
fprintf('Hardware used: NO\n');
fprintf('Training performed: NO\n');
fprintf('Original Dataset modified: NO\n');
fprintf('Leakage threshold: %.3f\n',similarityThreshold);
fprintf('Dataset:\n%s\n',datasetFile);
fprintf('Leakage pairs:\n%s\n',leakageFile);
fprintf('Output:\n%s\n',outputDatasetFile);
fprintf('============================================================\n');

%% =========================================================
% Load Dataset and leakage pairs
% ==========================================================

D = load(datasetFile);

requiredFields = {
    'realIQ'
    'captureOrder'
    'sessionID'
    'split'
    'txUnit'
    'modulationName'
    'className'
    'classLabel'
    'capturedOk'
};

for k = 1:numel(requiredFields)
    assert(isfield(D,requiredFields{k}), ...
        'Required Dataset field is missing: %s',requiredFields{k});
end

N = size(D.realIQ,1);
assert(N == 31800,'Expected 31,800 captures.');

originalSplit = lower(string(D.split(:)));
txUnit = string(D.txUnit(:));
modulationName = string(D.modulationName(:));
className = string(D.className(:));
captureOrder = double(D.captureOrder(:));

P = readtable(leakageFile,'TextType','string');

requiredPairColumns = {
    'QueryDatasetRow'
    'ReferenceDatasetRow'
    'Similarity'
};

for k = 1:numel(requiredPairColumns)
    assert(ismember(requiredPairColumns{k},P.Properties.VariableNames), ...
        'Required leakage column is missing: %s',requiredPairColumns{k});
end

P.Similarity = double(P.Similarity);
severePairs = P(P.Similarity >= similarityThreshold,:);

fprintf('\nLeakage pairs at or above %.3f: %d\n', ...
    similarityThreshold,height(severePairs));

%% =========================================================
% Build connected components with union-find
% ==========================================================

parent = (1:N).';
rankValue = zeros(N,1,'uint16');

for pairIndex = 1:height(severePairs)
    rowA = double(severePairs.QueryDatasetRow(pairIndex));
    rowB = double(severePairs.ReferenceDatasetRow(pairIndex));

    assert(rowA >= 1 && rowA <= N && rowB >= 1 && rowB <= N, ...
        'Leakage pair contains an invalid Dataset row.');

    rootA = findRoot(parent,rowA);
    rootB = findRoot(parent,rowB);

    if rootA ~= rootB
        if rankValue(rootA) < rankValue(rootB)
            parent(rootA) = rootB;
        elseif rankValue(rootA) > rankValue(rootB)
            parent(rootB) = rootA;
        else
            parent(rootB) = rootA;
            rankValue(rootA) = rankValue(rootA) + 1;
        end
    end
end

% Compress all paths.
for rowIndex = 1:N
    parent(rowIndex) = findRoot(parent,rowIndex);
end

[uniqueRoots,~,componentID] = unique(parent,'stable');
numberOfComponents = numel(uniqueRoots);
componentSize = accumarray(componentID,1,[numberOfComponents 1]);

fprintf('Connected components: %d\n',numberOfComponents);
fprintf('Multi-row leakage components: %d\n',sum(componentSize > 1));
fprintf('Largest component: %d rows\n',max(componentSize));

%% =========================================================
% Validate component consistency
% ==========================================================

componentModulation = strings(numberOfComponents,1);
componentTx = strings(numberOfComponents,1);
componentClass = strings(numberOfComponents,1);
componentMinimumOrder = zeros(numberOfComponents,1);
componentOriginalTrain = zeros(numberOfComponents,1);
componentOriginalValidation = zeros(numberOfComponents,1);
componentOriginalTest = zeros(numberOfComponents,1);

for componentIndex = 1:numberOfComponents
    rows = find(componentID == componentIndex);

    modValues = unique(modulationName(rows));
    txValues = unique(txUnit(rows));
    classValues = unique(className(rows));

    assert(numel(modValues) == 1, ...
        'A leakage component contains multiple modulations.');
    assert(numel(txValues) == 1, ...
        'A leakage component contains multiple TX units.');
    assert(numel(classValues) == 1, ...
        'A leakage component contains multiple classes.');

    componentModulation(componentIndex) = modValues;
    componentTx(componentIndex) = txValues;
    componentClass(componentIndex) = classValues;
    componentMinimumOrder(componentIndex) = min(captureOrder(rows));
    componentOriginalTrain(componentIndex) = sum(originalSplit(rows)=="train");
    componentOriginalValidation(componentIndex) = ...
        sum(originalSplit(rows)=="validation" | originalSplit(rows)=="val");
    componentOriginalTest(componentIndex) = sum(originalSplit(rows)=="test");
end

%% =========================================================
% Assign complete components to clean splits
% ==========================================================

cleanSplit = strings(N,1);
componentAssignedSplit = strings(numberOfComponents,1);
uniqueModulations = unique(modulationName);
uniqueTxUnits = unique(txUnit);
assignmentRows = cell(numel(uniqueModulations)*numel(uniqueTxUnits),1);
assignmentRowIndex = 0;

for modulationIndex = 1:numel(uniqueModulations)
    currentModulation = uniqueModulations(modulationIndex);

    for txIndex = 1:numel(uniqueTxUnits)
        currentTx = uniqueTxUnits(txIndex);

        combinationComponents = find( ...
            componentModulation == currentModulation & ...
            componentTx == currentTx);

        combinationRows = find( ...
            modulationName == currentModulation & ...
            txUnit == currentTx);

        assert(numel(combinationRows)==530, ...
            'Expected 530 captures for %s x %s.', ...
            currentModulation,currentTx);

        % Larger connected groups are assigned first. Ties use time order.
        sortingMatrix = [ ...
            -double(componentSize(combinationComponents)), ...
            componentMinimumOrder(combinationComponents)];
        [~,sortPosition] = sortrows(sortingMatrix,[1 2]);
        combinationComponents = combinationComponents(sortPosition);

        assignedCounts = zeros(1,3);

        for localIndex = 1:numel(combinationComponents)
            componentIndex = combinationComponents(localIndex);
            groupSize = double(componentSize(componentIndex));

            deficits = targetPerCombination - assignedCounts;
            feasible = deficits >= groupSize;

            originalCounts = [ ...
                componentOriginalTrain(componentIndex), ...
                componentOriginalValidation(componentIndex), ...
                componentOriginalTest(componentIndex)];

            if any(feasible)
                candidateScores = deficits;
                candidateScores(~feasible) = -Inf;

                % Prefer the original majority split when capacity permits.
                candidateScores = candidateScores + ...
                    0.001 * originalCounts;
                [~,selectedSplit] = max(candidateScores);
            else
                % If exact capacity is impossible, choose least overflow.
                projectedOverflow = max(0, ...
                    assignedCounts + groupSize - targetPerCombination);
                [~,selectedSplit] = min(projectedOverflow);
            end

            componentAssignedSplit(componentIndex) = ...
                splitLabels(selectedSplit);
            rows = find(componentID == componentIndex);
            cleanSplit(rows) = splitLabels(selectedSplit);
            assignedCounts(selectedSplit) = ...
                assignedCounts(selectedSplit) + groupSize;
        end

        assignmentRowIndex = assignmentRowIndex + 1;
        assignmentRows{assignmentRowIndex} = table( ...
            currentModulation,currentTx, ...
            assignedCounts(1),assignedCounts(2),assignedCounts(3), ...
            assignedCounts(1)-targetPerCombination(1), ...
            assignedCounts(2)-targetPerCombination(2), ...
            assignedCounts(3)-targetPerCombination(3), ...
            'VariableNames', { ...
                'ModulationName','TxUnit','TrainCount', ...
                'ValidationCount','TestCount','TrainDifference', ...
                'ValidationDifference','TestDifference'});
    end
end

assignmentSummary = vertcat(assignmentRows{1:assignmentRowIndex});

assert(all(cleanSplit ~= ""),'One or more rows were not assigned.');

%% =========================================================
% Verify known severe pairs no longer cross splits
% ==========================================================

queryRows = double(severePairs.QueryDatasetRow);
referenceRows = double(severePairs.ReferenceDatasetRow);
knownPairStillCrossSplit = ...
    cleanSplit(queryRows) ~= cleanSplit(referenceRows);

remainingKnownCrossSplitPairs = sum(knownPairStillCrossSplit);

fprintf('\nKnown severe pairs remaining across clean splits: %d\n', ...
    remainingKnownCrossSplitPairs);

assert(remainingKnownCrossSplitPairs == 0, ...
    'The clean split still separates known severe pairs.');

%% =========================================================
% Build distribution summaries
% ==========================================================

trainMask = cleanSplit == "train";
validationMask = cleanSplit == "validation";
testMask = cleanSplit == "test";

cleanSplitSummary = table( ...
    ["train";"validation";"test"], ...
    [sum(trainMask);sum(validationMask);sum(testMask)], ...
    [21000;5400;5400], ...
    [sum(trainMask)-21000;sum(validationMask)-5400;sum(testMask)-5400], ...
    'VariableNames', { ...
        'Split','ActualCount','TargetCount','Difference'});

metadataTable = table( ...
    (1:N).',captureOrder,originalSplit,cleanSplit, ...
    txUnit,modulationName,className,componentID, ...
    componentSize(componentID), ...
    'VariableNames', { ...
        'DatasetRow','CaptureOrder','OriginalSplit','CleanSplit', ...
        'TxUnit','ModulationName','ClassName','LeakageComponentID', ...
        'LeakageComponentSize'});

metadataTable.SplitChanged = ...
    metadataTable.OriginalSplit ~= metadataTable.CleanSplit;

changedRows = metadataTable(metadataTable.SplitChanged,:);

% Distribution by clean split, modulation and TX.
[groupID,groupSplit,groupMod,groupTx] = findgroups( ...
    metadataTable.CleanSplit, ...
    metadataTable.ModulationName, ...
    metadataTable.TxUnit);
groupCount = splitapply(@numel,metadataTable.DatasetRow,groupID);
distributionByCombination = table( ...
    groupSplit,groupMod,groupTx,groupCount, ...
    'VariableNames', { ...
        'CleanSplit','ModulationName','TxUnit','Count'});

% Distribution by class.
[classGroup,classSplit,classValue] = findgroups( ...
    metadataTable.CleanSplit,metadataTable.ClassName);
classCount = splitapply(@numel,metadataTable.DatasetRow,classGroup);
distributionByClass = table(classSplit,classValue,classCount, ...
    'VariableNames',{'CleanSplit','ClassName','Count'});

%% =========================================================
% Save the new Dataset without overwriting the original
% ==========================================================

D.originalSplit = cellstr(originalSplit.');
D.cleanSplit = cellstr(cleanSplit.');
D.split = D.cleanSplit;
D.cleanSplitSimilarityThreshold = similarityThreshold;
D.cleanSplitKnownCrossSplitPairsRemaining = ...
    remainingKnownCrossSplitPairs;
D.cleanSplitComponentID = int32(componentID.');
D.cleanSplitChanged = uint8(metadataTable.SplitChanged.');

save(outputDatasetFile,'-struct','D','-v7.3');

writetable(cleanSplitSummary,fullfile(resultsFolder, ...
    'clean_split_summary.csv'));
writetable(assignmentSummary,fullfile(resultsFolder, ...
    'clean_assignment_by_modulation_tx.csv'));
writetable(distributionByCombination,fullfile(resultsFolder, ...
    'clean_distribution_by_split_modulation_tx.csv'));
writetable(distributionByClass,fullfile(resultsFolder, ...
    'clean_distribution_by_class.csv'));
writetable(changedRows,fullfile(resultsFolder, ...
    'changed_split_rows.csv'));
writetable(metadataTable,fullfile(resultsFolder, ...
    'clean_split_full_metadata.csv'));

componentTable = table( ...
    (1:numberOfComponents).',componentSize,componentModulation, ...
    componentTx,componentClass,componentMinimumOrder, ...
    componentAssignedSplit,componentOriginalTrain, ...
    componentOriginalValidation,componentOriginalTest, ...
    'VariableNames', { ...
        'LeakageComponentID','ComponentSize','ModulationName', ...
        'TxUnit','ClassName','MinimumCaptureOrder', ...
        'AssignedCleanSplit','OriginalTrainCount', ...
        'OriginalValidationCount','OriginalTestCount'});

writetable(componentTable,fullfile(resultsFolder, ...
    'leakage_components.csv'));

save(fullfile(resultsFolder,'clean_split_results.mat'), ...
    'cleanSplit','originalSplit','componentID','componentTable', ...
    'cleanSplitSummary','assignmentSummary','metadataTable', ...
    'distributionByCombination','distributionByClass', ...
    'similarityThreshold','-v7.3');

%% =========================================================
% Final output
% ==========================================================

fprintf('\nCLEAN SPLIT SUMMARY\n');
fprintf('===================\n');
disp(cleanSplitSummary);

fprintf('\nChanged split rows: %d\n',height(changedRows));
fprintf('Known severe cross-split pairs remaining: %d\n', ...
    remainingKnownCrossSplitPairs);
fprintf('Maximum absolute combination-count difference: %d\n', ...
    max(abs([assignmentSummary.TrainDifference; ...
             assignmentSummary.ValidationDifference; ...
             assignmentSummary.TestDifference])));

fprintf('\n============================================================\n');
fprintf('CLEAN SPLIT V2 CREATED\n');
fprintf('============================================================\n');
fprintf('All captures preserved: %d / %d\n',N,N);
fprintf('Original Dataset modified: NO\n');
fprintf('Known pairs >= %.3f across clean splits: %d\n', ...
    similarityThreshold,remainingKnownCrossSplitPairs);
fprintf('Clean Dataset:\n%s\n',outputDatasetFile);
fprintf('Results folder:\n%s\n',resultsFolder);
fprintf('============================================================\n');

%% =========================================================
% Local helper: union-find root lookup
% ==========================================================

function root = findRoot(parent,node)
    root = node;
    while parent(root) ~= root
        root = parent(root);
    end
end
