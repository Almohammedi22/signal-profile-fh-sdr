%% inspect_temporal_similarity_31800.m
% OFFLINE temporal similarity and cross-split leakage inspection.
clear; clc; close all;
rng(42,'twister');

projectFolder = pwd;
datasetFile = fullfile(projectFolder,'results_real_dataset_31800', ...
    'dataset_31800_current','real_sdr_dataset_31800.mat');
resultsFolder = fullfile(projectFolder,'results_dataset_31800_temporal_inspection');
assert(isfile(datasetFile),'Dataset file was not found.');
if ~isfolder(resultsFolder), mkdir(resultsFolder); end

fprintf('\n============================================================\n');
fprintf('TEMPORAL SIMILARITY INSPECTION - SDR DATASET 31800\n');
fprintf('============================================================\n');
fprintf('Mode: OFFLINE DATASET INSPECTION\nHardware used: NO\nTraining performed: NO\n');
fprintf('Dataset:\n%s\n',datasetFile);

D = load(datasetFile,'realIQ','captureOrder','sessionID','split','txUnit', ...
    'modulationName','className','classLabel','capturedOk','sampleRate');
N = size(D.realIQ,1);
assert(N == 31800 && size(D.realIQ,2) == 256 && size(D.realIQ,3) == 2, ...
    'Unexpected Dataset dimensions.');

captureOrder = double(D.captureOrder(:));
sessionID = double(D.sessionID(:));
splitName = lower(string(D.split(:)));
txUnit = string(D.txUnit(:));
modulationName = string(D.modulationName(:));
className = string(D.className(:));
classLabel = double(D.classLabel(:));
capturedOk = logical(D.capturedOk(:));
trainMask = splitName == "train";
validationMask = splitName == "validation" | splitName == "val";
testMask = splitName == "test";

uniqueOrder = unique(captureOrder);
missingOrder = setdiff((1:N).',uniqueOrder);
orderPassed = numel(uniqueOrder)==N && isempty(missingOrder) && ...
    all(captureOrder>=1 & captureOrder<=N);
fprintf('\nCAPTURE-ORDER INTEGRITY\n');
fprintf('Unique: %d | Missing: %d | Passed: %d\n',numel(uniqueOrder),numel(missingOrder),orderPassed);

I = single(D.realIQ(:,:,1));
Q = single(D.realIQ(:,:,2));
x = complex(I,Q);
x = x - mean(x,2);
rmsCentered = sqrt(mean(abs(x).^2,2));
x = x ./ max(rmsCentered,single(1e-6));
x = x ./ max(sqrt(sum(abs(x).^2,2)),single(1e-6));

[sortedOrder,sortIndex] = sort(captureOrder);
sortedX = x(sortIndex,:);
sortedTx = txUnit(sortIndex);
sortedMod = modulationName(sortIndex);
sortedSplit = splitName(sortIndex);

adjacentSimilarity = nan(N,1,'single');
adjacentSimilarity(2:end) = abs(sum(conj(sortedX(1:end-1,:)).*sortedX(2:end,:),2));
validAdjacent = adjacentSimilarity(isfinite(adjacentSimilarity));
fprintf('\nADJACENT TEMPORAL SIMILARITY\n');
fprintf('Mean %.6f | Median %.6f | P99 %.6f\n', ...
    mean(validAdjacent),median(validAdjacent),prctile(validAdjacent,99));
fprintf('>=0.980: %d | >=0.995: %d | >=0.999: %d\n', ...
    sum(validAdjacent>=0.980),sum(validAdjacent>=0.995),sum(validAdjacent>=0.999));

previousSameGroupSimilarity = nan(N,1,'single');
previousSameGroupPosition = zeros(N,1);
keys = sortedTx + "|" + sortedMod;
uniqueKeys = unique(keys,'stable');
for k = 1:numel(uniqueKeys)
    rows = find(keys == uniqueKeys(k));
    if numel(rows) < 2, continue; end
    cur = rows(2:end); prev = rows(1:end-1);
    previousSameGroupSimilarity(cur) = abs(sum(conj(sortedX(prev,:)).*sortedX(cur,:),2));
    previousSameGroupPosition(cur) = prev;
end
validSame = previousSameGroupSimilarity(isfinite(previousSameGroupSimilarity));
fprintf('\nSAME TX/MODULATION TEMPORAL SIMILARITY\n');
fprintf('Mean %.6f | Median %.6f | P99 %.6f\n', ...
    mean(validSame),median(validSame),prctile(validSame,99));
fprintf('>=0.980: %d | >=0.995: %d | >=0.999: %d\n', ...
    sum(validSame>=0.980),sum(validSame>=0.995),sum(validSame>=0.999));

trainIdx = find(trainMask); valIdx = find(validationMask); testIdx = find(testMask);
blockSize = 256;
fprintf('\nCROSS-SPLIT SIMILARITY SEARCH\n');
fprintf('Train vs Validation...\n');
tv = findCrossSplitMaximums(x,trainIdx,valIdx,blockSize);
fprintf('Train vs Test...\n');
tt = findCrossSplitMaximums(x,trainIdx,testIdx,blockSize);
fprintf('Validation vs Test...\n');
vt = findCrossSplitMaximums(x,valIdx,testIdx,blockSize);

comparison = ["Train_vs_Validation";"Train_vs_Test";"Validation_vs_Test"];
R = {tv;tt;vt};
summaryRows = cell(3,1);
pairRows = cell(3,1);
for k = 1:3
    s = double(R{k}.MaximumSimilarity);
    summaryRows{k} = table(comparison(k),numel(s),mean(s),median(s), ...
        prctile(s,95),prctile(s,99),max(s),sum(s>=0.980),sum(s>=0.995),sum(s>=0.999), ...
        'VariableNames',{'Comparison','QueryCount','MeanMaximumSimilarity', ...
        'MedianMaximumSimilarity','P95','P99','MaximumSimilarity', ...
        'CountAbove0980','CountAbove0995','CountAbove0999'});
    m = s >= 0.980;
    q = R{k}.QueryDatasetRow(m); ref = R{k}.ReferenceDatasetRow(m);
    pairRows{k} = table(repmat(comparison(k),sum(m),1),q,ref,captureOrder(q),captureOrder(ref), ...
        splitName(q),splitName(ref),txUnit(q),txUnit(ref),modulationName(q), ...
        modulationName(ref),className(q),className(ref),single(s(m)), ...
        'VariableNames',{'Comparison','QueryDatasetRow','ReferenceDatasetRow', ...
        'QueryCaptureOrder','ReferenceCaptureOrder','QuerySplit','ReferenceSplit', ...
        'QueryTx','ReferenceTx','QueryModulation','ReferenceModulation', ...
        'QueryClass','ReferenceClass','Similarity'});
end
crossSplitSummary = vertcat(summaryRows{:});
nonempty = ~cellfun(@isempty,pairRows);
if any(nonempty), highPairs = vertcat(pairRows{nonempty}); else, highPairs = table(); end
if ~isempty(highPairs), highPairs = sortrows(highPairs,'Similarity','descend'); end
fprintf('\nCROSS-SPLIT SUMMARY\n'); disp(crossSplitSummary);

captureRMS = sqrt(mean(I.^2+Q.^2,2));
capturePeak = max(sqrt(I.^2+Q.^2),[],2);
meanI = mean(I,2); meanQ = mean(Q,2);
dcPct = 100*sqrt(meanI.^2+meanQ.^2)./max(captureRMS,single(1e-6));
powerI = mean(I.^2,2); powerQ = mean(Q.^2,2);
iqRatioDB = 10*log10(max(powerI,single(1e-12))./max(powerQ,single(1e-12)));

restartOrder = 26883; w = 500;
pre = captureOrder >= restartOrder-w & captureOrder < restartOrder;
post = captureOrder >= restartOrder & captureOrder < restartOrder+w;
metric = ["CaptureRMS";"CapturePeak";"NormalizedDCPct";"IQPowerRatioDB"];
preMedian = [median(captureRMS(pre));median(capturePeak(pre));median(dcPct(pre));median(iqRatioDB(pre))];
postMedian = [median(captureRMS(post));median(capturePeak(post));median(dcPct(post));median(iqRatioDB(post))];
absoluteChange = postMedian-preMedian;
relativeChangePct = 100*absoluteChange./max(abs(preMedian),1e-9);
restartComparison = table(metric,preMedian,postMedian,absoluteChange,relativeChangePct);
fprintf('\nPLUTO_03 RESTART COMPARISON\n'); disp(restartComparison);

maximumCrossSplitSimilarity = zeros(N,1,'single');
allR = {tv;tt;vt};
for k = 1:3
    for j = 1:numel(allR{k}.MaximumSimilarity)
        q = allR{k}.QueryDatasetRow(j); ref = allR{k}.ReferenceDatasetRow(j);
        s = allR{k}.MaximumSimilarity(j);
        maximumCrossSplitSimilarity(q) = max(maximumCrossSplitSimilarity(q),s);
        maximumCrossSplitSimilarity(ref) = max(maximumCrossSplitSimilarity(ref),s);
    end
end
qualityFlag = repmat("Good",N,1);
qualityFlag(maximumCrossSplitSimilarity>=0.980) = "Review";
qualityFlag(maximumCrossSplitSimilarity>=0.995) = "SevereReview";
[g,names] = findgroups(qualityFlag);
counts = splitapply(@numel,qualityFlag,g);
temporalQualitySummary = table(names,counts,100*counts/N, ...
    'VariableNames',{'TemporalQualityFlag','Count','Percentage'});
fprintf('\nTEMPORAL QUALITY SUMMARY\n'); disp(temporalQualitySummary);

writetable(crossSplitSummary,fullfile(resultsFolder,'cross_split_similarity_summary.csv'));
writetable(highPairs,fullfile(resultsFolder,'cross_split_high_similarity_pairs.csv'));
writetable(restartComparison,fullfile(resultsFolder,'pluto03_restart_comparison.csv'));
writetable(temporalQualitySummary,fullfile(resultsFolder,'temporal_quality_summary.csv'));
save(fullfile(resultsFolder,'temporal_similarity_results.mat'), ...
    'crossSplitSummary','highPairs','restartComparison','temporalQualitySummary', ...
    'adjacentSimilarity','previousSameGroupSimilarity','maximumCrossSplitSimilarity', ...
    'qualityFlag','-v7.3');

nearExact = height(highPairs)>0 && any(highPairs.Similarity>=0.999);
passed = orderPassed && ~nearExact;
fprintf('\n============================================================\n');
fprintf('TEMPORAL SIMILARITY INSPECTION COMPLETED\n');
fprintf('============================================================\n');
fprintf('Capture-order integrity passed: %d\n',orderPassed);
fprintf('Adjacent pairs >= 0.995: %d\n',sum(validAdjacent>=0.995));
fprintf('Same-group pairs >= 0.995: %d\n',sum(validSame>=0.995));
fprintf('Cross-split pairs >= 0.980: %d\n',height(highPairs));
if isempty(highPairs)
    fprintf('Cross-split pairs >= 0.995: 0\nCross-split pairs >= 0.999: 0\n');
else
    fprintf('Cross-split pairs >= 0.995: %d\n',sum(highPairs.Similarity>=0.995));
    fprintf('Cross-split pairs >= 0.999: %d\n',sum(highPairs.Similarity>=0.999));
end
if passed, fprintf('TEMPORAL INSPECTION STATUS: PASSED\n');
else, fprintf('TEMPORAL INSPECTION STATUS: NEEDS REVIEW\n'); end
fprintf('Results folder:\n%s\n',resultsFolder);
