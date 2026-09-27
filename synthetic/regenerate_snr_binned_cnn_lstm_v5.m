%% regenerate_snr_binned_cnn_lstm_v5.m
% Regenerate SNR-binned test accuracy using the FINAL, independently
% retrained CNN-LSTM occupied-signal checkpoint (82.92% accuracy, the one
% reported in Table 6 / net_cnn_lstm_occ_v5.mat) — so Section 6.3 can
% report results consistent with the rest of the paper instead of the
% stale figures from the earlier 90.33% training run.
%
% Requires, in the current folder:
%   dataset_v5.mat            (X, y_state, test_idx, and a per-sample SNR field)
%   net_cnn_lstm_occ_v5.mat   (netCNNLSTMocc - the 82.92% trained network)
%
% SELF-CHECK: before trusting the SNR-binned breakdown, this script
% reclassifies the reconstructed occupied test set and confirms the
% resulting OVERALL accuracy matches the already-verified 82.92% figure
% to within rounding. If it does not match, the warning below will say
% so explicitly — stop and send me the mismatch (and, ideally,
% prepare_occupied_sequences_v5.m) rather than reporting the SNR numbers.

clear; clc; close all;

datasetFile = 'dataset_v5.mat';
networkFile = 'net_cnn_lstm_occ_v5.mat';

if ~isfile(datasetFile)
    error('Required file not found: %s', datasetFile);
end
if ~isfile(networkFile)
    error('Required file not found: %s', networkFile);
end

%% Load dataset and locate the SNR field

D = load(datasetFile);

if ~isfield(D, 'X') || ~isfield(D, 'y_state') || ~isfield(D, 'test_idx')
    error('dataset_v5.mat is missing X, y_state, or test_idx.');
end

X = single(D.X);
yState = double(D.y_state(:));
testIdx = double(D.test_idx(:));

snrCandidates = {'snr_db', 'SNR', 'snr', 'SNR_dB', 'snr_dB', 'rho', ...
    'SNRValues', 'snrValues', 'SNR_dB_'};
snrField = '';
for k = 1:numel(snrCandidates)
    if isfield(D, snrCandidates{k})
        snrField = snrCandidates{k};
        break;
    end
end

if isempty(snrField)
    fprintf('Could not find a recognized SNR field automatically.\n');
    fprintf('Fields present in %s:\n', datasetFile);
    disp(fieldnames(D));
    error(['Add the correct variable name to snrCandidates above (or set ' ...
           'snrField directly) and re-run.']);
end

fprintf('Using SNR field: %s\n', snrField);
snrAll = double(D.(snrField)(:));

if numel(snrAll) ~= size(X, 1)
    error(['SNR field length (%d) does not match number of samples in X ' ...
           '(%d). Check whether %s is per-sample.'], ...
           numel(snrAll), size(X, 1), snrField);
end

%% Determine the y_state code mapping (robust to 0- vs 1-based indexing)
% energy_threshold_sensing_v5.m documents 0=idle, 1=friendly, 2=adversarial,
% but dataset_v5.mat also ships a 'stateNames' field -- use it to confirm
% (or correct) that mapping rather than trusting the hardcoded guess,
% since this dataset turns out to carry several other label fields
% (y_system, y_fh_pattern, y_next_hop, ...) that are NOT the friendly/
% adversarial label and must not be confused with y_state.

if isfield(D, 'stateNames')
    stateNames = D.stateNames;
    if iscell(stateNames) || isstring(stateNames)
        stateNamesStr = string(stateNames(:));
    else
        stateNamesStr = string(cellstr(stateNames));
    end
    fprintf('stateNames found: %s\n', strjoin(stateNamesStr, ', '));

    friendlyIdx = find(strcmpi(stateNamesStr, 'friendly'), 1);
    adversarialIdx = find(strcmpi(stateNamesStr, 'adversarial'), 1);
    idleIdx = find(strcmpi(stateNamesStr, 'idle'), 1);

    if isempty(friendlyIdx) || isempty(adversarialIdx) || isempty(idleIdx)
        error(['stateNames does not contain the expected idle/friendly/' ...
               'adversarial labels (found: %s). Inspect dataset_v5.mat ' ...
               'and set idleCode/friendlyCode/adversarialCode manually.'], ...
               strjoin(stateNamesStr, ', '));
    end

    codesPresent = unique(yState);
    fprintf('Numeric codes present in y_state: %s\n', mat2str(codesPresent(:)'));

    if min(codesPresent) == 0
        % 0-based: stateNames{k} corresponds to code k-1
        idleCode = idleIdx - 1;
        friendlyCode = friendlyIdx - 1;
        adversarialCode = adversarialIdx - 1;
    else
        % 1-based: stateNames{k} corresponds to code k
        idleCode = idleIdx;
        friendlyCode = friendlyIdx;
        adversarialCode = adversarialIdx;
    end
else
    fprintf(['No stateNames field found; falling back to the documented ' ...
        'convention (0=idle, 1=friendly, 2=adversarial).\n']);
    idleCode = 0;
    friendlyCode = 1;
    adversarialCode = 2;
end

fprintf('Resolved codes -> idle=%d, friendly=%d, adversarial=%d\n', ...
    idleCode, friendlyCode, adversarialCode);

%% Reconstruct the occupied test subset
% Same filter as prepare_occupied_sequences_v5.m: restrict to the test
% split, then keep only occupied samples (y_state ~= idleCode), preserving
% the original test_idx order.

occupiedMask = yState(testIdx) ~= idleCode;
occTestIdx = testIdx(occupiedMask);

fprintf('Test samples (all states): %d\n', numel(testIdx));
fprintf('Occupied test samples: %d\n', numel(occTestIdx));

yTestState = yState(occTestIdx);
snrTest = snrAll(occTestIdx);

classNames = ["friendly"; "adversarial"];
YTestTrue = categorical(yTestState, [friendlyCode adversarialCode], cellstr(classNames));

%% Convert to the 2x256 cell format expected by the CNN-LSTM network

numTest = numel(occTestIdx);
XTest = cell(numTest, 1);

for i = 1:numTest
    sample = squeeze(X(occTestIdx(i), :, :));  % 256 x 2 (time x IQ)
    XTest{i} = single(sample.');                % 2 x 256 (IQ x time)
end

fprintf('Reconstructed sequence format: %s\n', mat2str(size(XTest{1})));

%% Load trained network and classify

net = load(networkFile);

if ~isfield(net, 'netCNNLSTMocc')
    error('net_cnn_lstm_occ_v5.mat does not contain netCNNLSTMocc.');
end

YPred = classify(net.netCNNLSTMocc, XTest, ...
    'MiniBatchSize', 256, ...
    'ExecutionEnvironment', 'auto');

YPred = YPred(:);

%% SELF-CHECK against the already-verified 82.92% result — do this first

overallAcc = mean(YPred == YTestTrue) * 100;

fprintf('\n============================================\n');
fprintf('SELF-CHECK\n');
fprintf('============================================\n');
fprintf('Reconstructed overall test accuracy: %.4f%%\n', overallAcc);
fprintf('Expected (Table 6 / cnn_lstm_occ_v5_results.csv): 82.9167%%\n');

if abs(overallAcc - 82.9167) > 0.05
    warning(['Reconstructed accuracy does NOT match the verified 82.92%% ' ...
        'result within tolerance. DO NOT report the SNR-binned numbers ' ...
        'below until this is resolved -- the test-set reconstruction or ' ...
        'the I/Q axis ordering is likely wrong. Send the mismatch (and ' ...
        'ideally prepare_occupied_sequences_v5.m) back for debugging.']);
else
    fprintf('MATCH -- reconstruction is consistent with the verified run.\n');
    fprintf('The SNR-binned breakdown below can be trusted.\n');
end

%% SNR-binned accuracy

edges = [-20 -10 0 10 25];
labels = {'[-20,-10)', '[-10,0)', '[0,10)', '[10,25]'};

nBins = numel(edges) - 1;
binN = zeros(nBins, 1);
binAcc = zeros(nBins, 1);
binFriendlyRecall = zeros(nBins, 1);
binAdversarialRecall = zeros(nBins, 1);

for b = 1:nBins
    if b < nBins
        inBin = snrTest >= edges(b) & snrTest < edges(b+1);
    else
        inBin = snrTest >= edges(b) & snrTest <= edges(b+1);  % inclusive top edge
    end

    binN(b) = sum(inBin);

    if binN(b) == 0
        binAcc(b) = NaN;
        binFriendlyRecall(b) = NaN;
        binAdversarialRecall(b) = NaN;
        continue;
    end

    yTrueBin = YTestTrue(inBin);
    yPredBin = YPred(inBin);

    binAcc(b) = mean(yTrueBin == yPredBin) * 100;

    friendlyMask = yTrueBin == "friendly";
    adversarialMask = yTrueBin == "adversarial";

    if any(friendlyMask)
        binFriendlyRecall(b) = mean(yPredBin(friendlyMask) == "friendly") * 100;
    else
        binFriendlyRecall(b) = NaN;
    end

    if any(adversarialMask)
        binAdversarialRecall(b) = mean(yPredBin(adversarialMask) == "adversarial") * 100;
    else
        binAdversarialRecall(b) = NaN;
    end
end

fprintf('\n============================================\n');
fprintf('SNR-BINNED RESULTS -- FINAL CNN-LSTM CHECKPOINT (82.92%% run)\n');
fprintf('============================================\n');

resultsTable = table( ...
    labels(:), ...
    binN, ...
    binAcc, ...
    binFriendlyRecall, ...
    binAdversarialRecall, ...
    'VariableNames', { ...
        'SNRInterval', ...
        'NumTestSamples', ...
        'AccuracyPercent', ...
        'FriendlyRecallPercent', ...
        'AdversarialRecallPercent' ...
        });

disp(resultsTable);

writetable(resultsTable, 'cnn_lstm_occ_v5_snr_binned_results.csv');

%% Also save the per-sample predictions + SNR so this can be independently
%  re-verified row-level, same as every other result in this project.

perSampleTable = table( ...
    occTestIdx(:), ...
    snrTest(:), ...
    string(YTestTrue(:)), ...
    string(YPred(:)), ...
    double(YTestTrue(:) == YPred(:)), ...
    'VariableNames', { ...
        'OriginalDatasetIndex', ...
        'SNR_dB', ...
        'TrueLabel', ...
        'PredictedLabel', ...
        'Correct' ...
        });

writetable(perSampleTable, 'cnn_lstm_occ_v5_snr_binned_predictions.csv');

%% Plot

figure('Color', 'white');
bar(binAcc);
set(gca, 'XTickLabel', labels);
ylim([0 100]);
grid on;
xlabel('SNR interval (dB)');
ylabel('Test accuracy (%)');
title('CNN-LSTM (82.92% checkpoint) -- SNR-Binned Test Accuracy');

exportgraphics(gcf, 'cnn_lstm_occ_v5_snr_binned.png', 'Resolution', 200);

fprintf('\nSaved files:\n');
fprintf('- cnn_lstm_occ_v5_snr_binned_results.csv       (summary, 4 rows)\n');
fprintf('- cnn_lstm_occ_v5_snr_binned_predictions.csv   (per-sample, for verification)\n');
fprintf('- cnn_lstm_occ_v5_snr_binned.png\n');
