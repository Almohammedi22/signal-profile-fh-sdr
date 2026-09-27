function run_live_end_to_end_hardware_v1()
% RUN_LIVE_END_TO_END_HARDWARE_V1
% Fresh end-to-end hardware execution.
%
% Classification:
%   Hardware TX: YES
%   Hardware RX: YES
%   New training: NO
%   Reuse old capture: NO
%   Automatic post-decision retuning: NO (advisory only)
%
% Pipeline:
%   Pluto_02 TX -> Combiner -> Attenuator -> Pluto_03 RX
%   -> physical channel detection -> FH V6 -> DDQN
%   -> safety verification -> final advisory action

    clc;
    close all;

    trainingFolder = ...
        pwd;

    sdrFolder = ...
        fullfile(fileparts(pwd), ...
         'sdr_validation_package_v2', 'sdr_test');

    hardwareScript = fullfile( ...
        trainingFolder, ...
        'run_fh_v6_hardware_test.m');

    actionScript = fullfile( ...
        trainingFolder, ...
        'final_hardware_action_demo_v1.m');

    hardwareResultsRoot = fullfile( ...
        sdrFolder, ...
        'results', ...
        'fh_v6_hardware');

    finalResultsRoot = fullfile( ...
        trainingFolder, ...
        'results_final_hardware_action');

    if ~isfile(hardwareScript)
        error('Hardware script was not found: %s', hardwareScript);
    end

    if ~isfile(actionScript)
        error('Action script was not found: %s', actionScript);
    end

    cd(trainingFolder);

    fprintf('\n');
    fprintf('============================================\n');
    fprintf('LIVE END-TO-END HARDWARE PIPELINE V1\n');
    fprintf('============================================\n');
    fprintf('Mode: FRESH PHYSICAL TRANSMISSION\n');
    fprintf('Training: NO\n');
    fprintf('Hardware transmission: YES\n');
    fprintf('Hardware reception: YES\n');
    fprintf('Reuse old capture: NO\n');
    fprintf('Automatic final retuning: NO\n');
    fprintf('\nPhysical path:\n');
    fprintf('Pluto_02 TX -> Combiner -> Attenuator -> Pluto_03 RX\n');
    fprintf('TX URI: ip:192.168.2.1\n');
    fprintf('RX URI: ip:192.168.3.1\n');
    fprintf('============================================\n');

    oldHardwareRuns = listRunNames(hardwareResultsRoot);
    oldFinalRuns = listRunNames(finalResultsRoot);

    fprintf('\nStarting a fresh physical TX/RX run...\n\n');

    % The hardware script is isolated because it begins with clear.
    executeScriptIsolated(hardwareScript);

    newHardwareRuns = listRunNames(hardwareResultsRoot);
    createdHardwareRuns = setdiff( ...
        newHardwareRuns, oldHardwareRuns, 'stable');

    if isempty(createdHardwareRuns)
        error(['No new hardware run was created. ', ...
            'An old capture will not be reused.']);
    end

    freshHardwareFolder = newestNamedRun( ...
        hardwareResultsRoot, createdHardwareRuns);

    captureFile = fullfile( ...
        freshHardwareFolder, ...
        'fh_hardware_capture.mat');

    hardwareResultFile = fullfile( ...
        freshHardwareFolder, ...
        'fh_v6_hardware_results.mat');

    hardwareSummaryFile = fullfile( ...
        freshHardwareFolder, ...
        'fh_v6_hardware_summary.csv');

    requiredFreshFiles = {
        captureFile
        hardwareResultFile
        hardwareSummaryFile
    };

    for fileIndex = 1:numel(requiredFreshFiles)
        if ~isfile(requiredFreshFiles{fileIndex})
            error('Fresh hardware run is incomplete: %s', ...
                requiredFreshFiles{fileIndex});
        end
    end

    H = load(captureFile);
    trueChannels = double(H.true_channels(:));
    detectedChannels = double(H.detected_channels(:));
    separationDB = double(H.separation_db(:));

    if numel(trueChannels) ~= numel(detectedChannels)
        error('Fresh TX and RX channel counts do not match.');
    end

    freshDetectionAccuracy = mean( ...
        trueChannels == detectedChannels) * 100;

    fprintf('\n');
    fprintf('============================================\n');
    fprintf('FRESH HARDWARE RUN VERIFIED\n');
    fprintf('============================================\n');
    fprintf('New hardware folder:\n%s\n', freshHardwareFolder);
    fprintf('Physical hops transmitted: %d\n', ...
        numel(trueChannels));
    fprintf('Physical hops detected: %d\n', ...
        numel(detectedChannels));
    fprintf('Correct detections: %d / %d\n', ...
        sum(trueChannels == detectedChannels), ...
        numel(trueChannels));
    fprintf('Detection accuracy: %.2f%%\n', ...
        freshDetectionAccuracy);
    fprintf('Mean separation: %.2f dB\n', mean(separationDB));
    fprintf('Minimum separation: %.2f dB\n', min(separationDB));
    fprintf('Maximum separation: %.2f dB\n', max(separationDB));
    fprintf('============================================\n');

    if freshDetectionAccuracy < 90
        error(['Fresh physical detection accuracy is %.2f%%. ', ...
            'Inference stopped before the final action.'], ...
            freshDetectionAccuracy);
    end

    fprintf('\nRunning FH V6, DDQN, and final advisory action...\n\n');

    % This script always reads the newest hardware run. The fresh run was
    % just created and is verified again below.
    executeScriptIsolated(actionScript);

    newFinalRuns = listRunNames(finalResultsRoot);
    createdFinalRuns = setdiff( ...
        newFinalRuns, oldFinalRuns, 'stable');

    if isempty(createdFinalRuns)
        error('No new final-action run was created.');
    end

    freshFinalFolder = newestNamedRun( ...
        finalResultsRoot, createdFinalRuns);

    finalSummaryFile = fullfile( ...
        freshFinalFolder, ...
        'final_hardware_action_summary.csv');

    if ~isfile(finalSummaryFile)
        error('Final action summary was not created: %s', ...
            finalSummaryFile);
    end

    S = readtable(finalSummaryFile, 'TextType', 'string');

    usedHardwareFolder = string(S.HardwareRunFolder(1));

    if usedHardwareFolder ~= string(freshHardwareFolder)
        error(['Safety stop: the final action did not use the ', ...
            'fresh hardware capture.']);
    end

    fprintf('\n');
    fprintf('============================================\n');
    fprintf('LIVE END-TO-END TEST COMPLETED\n');
    fprintf('============================================\n');
    fprintf('New physical transmission executed: YES\n');
    fprintf('New physical reception executed: YES\n');
    fprintf('Old capture reused: NO\n');
    fprintf('Physical detection accuracy: %.2f%%\n', ...
        freshDetectionAccuracy);
    fprintf('True next hop: %.0f\n', S.TrueNextHop(1));
    fprintf('FH Top-1: %.0f\n', S.FHTop1(1));
    fprintf('FH Top-5: %s\n', S.FHTop5(1));
    fprintf('DDQN action rank: %.0f\n', ...
        S.DDQNActionRank(1));
    fprintf('DDQN selected channel: %.0f\n', ...
        S.DDQNSelectedChannel(1));
    fprintf('DDQN selected protected: %d\n', ...
        logical(S.DDQNSelectedProtected(1)));
    fprintf('Final action: %s\n', S.FinalAction(1));
    fprintf('Final channel: %.0f\n', S.FinalChannel(1));
    fprintf('Final frequency: %.6f MHz\n', ...
        S.FinalFrequencyMHz(1));
    fprintf('Confidence: %s\n', S.Confidence(1));
    fprintf('\nFresh hardware result:\n%s\n', ...
        freshHardwareFolder);
    fprintf('\nFresh final-action result:\n%s\n', ...
        freshFinalFolder);
    fprintf('\nNo automatic post-decision retuning was performed.\n');
    fprintf('============================================\n');

    winopen(freshFinalFolder);
end

function executeScriptIsolated(scriptPath)
% Execute a script that may contain clear without clearing the main scope.
    run(scriptPath);
end

function runNames = listRunNames(rootFolder)
% Return names of timestamped result folders.
    runNames = strings(0, 1);

    if ~isfolder(rootFolder)
        return;
    end

    runFolders = dir(fullfile(rootFolder, 'run_*'));
    runFolders = runFolders([runFolders.isdir]);

    if ~isempty(runFolders)
        runNames = string({runFolders.name}).';
    end
end

function newestFolder = newestNamedRun(rootFolder, runNames)
% Select the newest folder from a list of run folder names.
    if isempty(runNames)
        error('No candidate run folders were supplied.');
    end

    candidateFolders = dir(fullfile(rootFolder, 'run_*'));
    candidateFolders = candidateFolders([candidateFolders.isdir]);

    keepMask = ismember(string({candidateFolders.name}), runNames);
    candidateFolders = candidateFolders(keepMask);

    if isempty(candidateFolders)
        error('No matching run folder was found.');
    end

    [~, newestIndex] = max([candidateFolders.datenum]);

    newestFolder = fullfile( ...
        candidateFolders(newestIndex).folder, ...
        candidateFolders(newestIndex).name);
end
