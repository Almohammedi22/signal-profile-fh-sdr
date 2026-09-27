%% prepare_occupied_sequences_v5.m

clear;
clc;

datasetFile = 'dataset_v5.mat';

if ~isfile(datasetFile)
    error('Dataset file not found: %s', datasetFile);
end

D = load( ...
    datasetFile, ...
    'X', ...
    'y_state', ...
    'train_idx', ...
    'val_idx', ...
    'test_idx');

X = single(D.X);
yState = double(D.y_state(:));

trainIdx = double(D.train_idx(:));
valIdx = double(D.val_idx(:));
testIdx = double(D.test_idx(:));

% Occupied classes:
% 1 = friendly
% 2 = adversarial

trainIdxOcc = trainIdx(yState(trainIdx) ~= 0);
valIdxOcc = valIdx(yState(valIdx) ~= 0);
testIdxOcc = testIdx(yState(testIdx) ~= 0);

XTrain = cell(numel(trainIdxOcc), 1);
XValidation = cell(numel(valIdxOcc), 1);
XTest = cell(numel(testIdxOcc), 1);

for k = 1:numel(trainIdxOcc)
    sample = squeeze(X(trainIdxOcc(k), :, :));
    XTrain{k} = single(sample.');
end

for k = 1:numel(valIdxOcc)
    sample = squeeze(X(valIdxOcc(k), :, :));
    XValidation{k} = single(sample.');
end

for k = 1:numel(testIdxOcc)
    sample = squeeze(X(testIdxOcc(k), :, :));
    XTest{k} = single(sample.');
end

YTrain = categorical( ...
    yState(trainIdxOcc), ...
    [1 2], ...
    {'friendly', 'adversarial'});

YValidation = categorical( ...
    yState(valIdxOcc), ...
    [1 2], ...
    {'friendly', 'adversarial'});

YTest = categorical( ...
    yState(testIdxOcc), ...
    [1 2], ...
    {'friendly', 'adversarial'});

fprintf('\nOccupied sequence preparation completed.\n');
fprintf('Train: %d\n', numel(XTrain));
fprintf('Validation: %d\n', numel(XValidation));
fprintf('Test: %d\n', numel(XTest));

fprintf('\nTraining distribution:\n');
disp(countcats(YTrain));

fprintf('Validation distribution:\n');
disp(countcats(YValidation));

fprintf('Test distribution:\n');
disp(countcats(YTest));

save( ...
    'occupied_sequences_v5.mat', ...
    'XTrain', ...
    'YTrain', ...
    'XValidation', ...
    'YValidation', ...
    'XTest', ...
    'YTest', ...
    'trainIdxOcc', ...
    'valIdxOcc', ...
    'testIdxOcc', ...
    '-v7.3');

fprintf('\nSaved occupied_sequences_v5.mat\n');