%% capture_real_dataset_31800_round_robin.m
% HARDWARE DATA COLLECTION ONLY
% Collects 31,800 physically received I/Q captures from four PlutoSDRs.
% No CNN training, FH, QCQP, DDQN, or Action Layer is executed here.

clear;
clc;
close all;

%% Collection control
% Set to 0 to continue until all 31,800 captures are successful.
% Set to a positive number, such as 600, to collect only that many new
% successful captures in the current run, then resume later.
maxNewCaptures = 0;

%% Resolve folders automatically
projectFolder = fileparts(mfilename('fullpath'));
if isempty(projectFolder)
    projectFolder = pwd;
end

pythonScript = fullfile(projectFolder, 'capture_real_dataset_31800_round_robin.py');
outputFolder = fullfile(projectFolder, 'results_real_dataset_31800', ...
    'dataset_31800_current');
pythonExe = fullfile(getenv('LOCALAPPDATA'), 'Programs', 'Python', ...
    'Python311', 'python.exe');
libiioFolder = fullfile('C:', 'ProgramData', 'MATLAB', ...
    'SupportPackages', 'R2026a', '3P.instrset', ...
    'libiio.instrset', 'win64', 'Windows-VS-2022-x64');

cd(projectFolder);

%% Display operation type
disp(' ');
disp('============================================================');
disp('REAL SDR DATASET - 31,800 HARDWARE CAPTURES');
disp('============================================================');
disp('STEP TYPE: HARDWARE DATA COLLECTION');
disp('Training: NO');
disp('Physical transmission: YES');
disp('Physical reception: YES');
disp('Saved data: Pluto_03 physically received I/Q');
disp('AI-generated waveforms: NO');
disp('Conventional DSP waveforms: YES');
disp('FH / QCQP / DDQN / Action: NOT USED');
disp('TX: Pluto_02, Pluto_04, Pluto_05');
disp('RX: Pluto_03');
disp('Plan: 20 modulations x 3 TX x 530 = 31,800');
disp('Train: 21,000 | Validation: 5,400 | Test: 5,400');
fprintf('New successful captures requested this run: %d\n', ...
    maxNewCaptures);
disp(['Project folder: ' projectFolder]);
disp(['Output folder: ' outputFolder]);
disp('============================================================');

%% Validate required files
if ~isfile(pythonExe)
    error('Python executable was not found: %s', pythonExe);
end
if ~isfile(pythonScript)
    error('Python collector was not found: %s', pythonScript);
end
if ~isfolder(libiioFolder)
    error('libiio folder was not found: %s', libiioFolder);
end
if ~isfile(fullfile(libiioFolder, 'libiio.dll'))
    error('libiio.dll was not found inside: %s', libiioFolder);
end
if ~isfolder(outputFolder)
    [folderStatus, folderMessage] = mkdir(outputFolder);
    if ~folderStatus
        error('Could not create output folder: %s', folderMessage);
    end
end

%% Check the four Pluto network addresses
ipAddresses = {
    '192.168.2.1'
    '192.168.3.1'
    '192.168.4.1'
    '192.168.5.1'
};

disp(' ');
disp('Checking Pluto network connectivity...');
for index = 1:numel(ipAddresses)
    command = sprintf('ping -n 1 %s', ipAddresses{index});
    [pingStatus, ~] = system(command);
    fprintf('%s reachable = %d\n', ipAddresses{index}, pingStatus == 0);
    if pingStatus ~= 0
        error('Pluto address is not reachable: %s', ipAddresses{index});
    end
end

%% Validate Python syntax
compileCommand = sprintf('"%s" -m py_compile "%s"', ...
    pythonExe, pythonScript);
[compileStatus, compileOutput] = system(compileCommand);
if compileStatus ~= 0
    disp(compileOutput);
    error('The Python collector contains a syntax error.');
end
disp('Python syntax check: PASS');

%% Start or resume physical collection
command = sprintf([ ...
    '"%s" -u "%s" --libiio-folder "%s" --output-dir "%s" ' ...
    '--max-new-captures %d'], ...
    pythonExe, pythonScript, libiioFolder, outputFolder, ...
    maxNewCaptures);

disp(' ');
disp('Starting or resuming physical TX/RX collection now...');
disp('One transmitter will be active at a time.');
disp('Press Ctrl+C only if you need to stop; checkpointing is enabled.');
disp(' ');
[collectorStatus, collectorOutput] = system(command, '-echo');

%% Read collection status
manifestFile = fullfile(outputFolder, 'manifest_31800.json');
groundTruthFile = fullfile(outputFolder, 'ground_truth_31800.csv');
iqFile = fullfile(outputFolder, 'real_iq_31800.npy');
okFile = fullfile(outputFolder, 'captured_ok_31800.npy');
finalMatFile = fullfile(outputFolder, 'real_sdr_dataset_31800.mat');
planFile = fullfile(outputFolder, 'capture_plan_31800.json');

requiredOutputFiles = {
    manifestFile
    groundTruthFile
    iqFile
    okFile
    planFile
};
for index = 1:numel(requiredOutputFiles)
    if ~isfile(requiredOutputFiles{index})
        disp(collectorOutput);
        error('Missing collection output: %s', requiredOutputFiles{index});
    end
end

manifest = jsondecode(fileread(manifestFile));
groundTruth = readtable(groundTruthFile, 'TextType', 'string');

successfulRows = groundTruth(groundTruth.captured_ok ~= 0, :);

disp(' ');
disp('============================================================');
disp('REAL HARDWARE DATASET STATUS');
disp('============================================================');
fprintf('Successful captures: %d / 31800\n', ...
    manifest.successful_captures);
fprintf('Remaining captures: %d\n', manifest.failed_or_pending);
fprintf('Recorded rows: %d\n', manifest.recorded_rows);
fprintf('Complete: %d\n', manifest.complete);
if ~isempty(successfulRows)
    fprintf('Mean capture/noise: %.2f dB\n', ...
        mean(successfulRows.capture_to_noise_db, 'omitnan'));
    fprintf('Mean window/noise: %.2f dB\n', ...
        mean(successfulRows.window_to_noise_db, 'omitnan'));
    fprintf('Clipping rows: %d\n', ...
        sum(successfulRows.clipping_detected ~= 0));
end
fprintf('Train target: %d\n', manifest.train_count);
fprintf('Validation target: %d\n', manifest.validation_count);
fprintf('Test target: %d\n', manifest.test_count);
disp(['I/Q checkpoint: ' iqFile]);
disp(['Ground truth: ' groundTruthFile]);
disp(['Manifest: ' manifestFile]);
if isfile(finalMatFile)
    disp(['Final MATLAB dataset: ' finalMatFile]);
else
    disp('Final MATLAB dataset will be created after 31,800/31,800.');
end
disp('============================================================');

if collectorStatus ~= 0 && ~manifest.complete
    warning(['Collection is incomplete. Run capture_real_dataset_31800 ' ...
        'again to resume failed or pending rows.']);
end

winopen(outputFolder);
