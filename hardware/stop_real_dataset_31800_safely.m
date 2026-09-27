%% stop_real_dataset_31800_safely.m
% Run this from a second MATLAB window, or create the same file manually.
projectFolder = fileparts(mfilename('fullpath'));
stopFile = fullfile(projectFolder, ...
    'results_real_dataset_31800', ...
    'dataset_31800_current', ...
    'STOP_AFTER_CURRENT_CAPTURE.txt');
outputFolder = fileparts(stopFile);
if ~isfolder(outputFolder)
    error('The active dataset output folder does not exist.');
end
fileID = fopen(stopFile,'w');
if fileID == -1
    error('Could not create the safe-stop request file.');
end
fprintf(fileID,'Stop requested at %s\n',char(datetime('now')));
fclose(fileID);
fprintf('Safe stop requested. Current capture will finish and save.\n');
fprintf('Stop file: %s\n',stopFile);
