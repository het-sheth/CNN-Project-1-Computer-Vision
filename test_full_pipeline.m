% test_full_cnn_pipeline.m
% This script performs an integration test on the entire 18-layer CNN.
% It runs a single image through the full pipeline, checking the output
% of every single layer against the known correct results from the
% debugging file.

clear;
clc;
fprintf('--- Starting Full 18-Layer CNN Integration Test ---\n\n');

% --- 1. Load All Necessary Data ---
try
    load 'debuggingTest.mat';
    load 'CNNparameters.mat';
    fprintf('Successfully loaded testing and parameter files.\n');
catch
    error('FAILED: Could not find debuggingTest.mat or CNNparameters.mat.');
end

% --- 2. Initialize and Run the Pipeline Step-by-Step ---
all_tests_passed = true;
current_output = imrgb; % Start with the raw input image

try
    % --- Layer 1: Normalization ---
    fprintf('\nTesting Layer 1 (imnormalize)... ');
    current_output = apply_imnormalize(current_output);
    if max(abs(current_output - layerResults{1}), [], 'all') < 1e-6
        fprintf('SUCCESS!\n');
    else
        fprintf('FAILURE.\n'); all_tests_passed = false;
    end

    % --- Layer 2: Convolution ---
    fprintf('Testing Layer 2 (convolve)... ');
    current_output = apply_convolve(current_output, filterbanks{2}, biasvectors{2});
    if max(abs(current_output - layerResults{2}), [], 'all') < 1e-6
        fprintf('SUCCESS!\n');
    else
        fprintf('FAILURE.\n'); all_tests_passed = false;
    end
    
    % --- Layer 3: ReLU ---
    fprintf('Testing Layer 3 (relu)... ');
    current_output = apply_relu(current_output);
    if max(abs(current_output - layerResults{3}), [], 'all') < 1e-6
        fprintf('SUCCESS!\n');
    else
        fprintf('FAILURE.\n'); all_tests_passed = false;
    end

    % --- Layers 4-18 (Continue the pattern) ---
    % Layer 4
    fprintf('Testing Layer 4 (convolve)... ');
    current_output = apply_convolve(current_output, filterbanks{4}, biasvectors{4});
    if max(abs(current_output - layerResults{4}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 5
    fprintf('Testing Layer 5 (relu)... ');
    current_output = apply_relu(current_output);
    if max(abs(current_output - layerResults{5}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 6
    fprintf('Testing Layer 6 (maxpool)... ');
    current_output = apply_maxpool(current_output);
    if max(abs(current_output - layerResults{6}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 7
    fprintf('Testing Layer 7 (convolve)... ');
    current_output = apply_convolve(current_output, filterbanks{7}, biasvectors{7});
    if max(abs(current_output - layerResults{7}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 8
    fprintf('Testing Layer 8 (relu)... ');
    current_output = apply_relu(current_output);
    if max(abs(current_output - layerResults{8}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 9
    fprintf('Testing Layer 9 (convolve)... ');
    current_output = apply_convolve(current_output, filterbanks{9}, biasvectors{9});
    if max(abs(current_output - layerResults{9}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 10
    fprintf('Testing Layer 10 (relu)... ');
    current_output = apply_relu(current_output);
    if max(abs(current_output - layerResults{10}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 11
    fprintf('Testing Layer 11 (maxpool)... ');
    current_output = apply_maxpool(current_output);
    if max(abs(current_output - layerResults{11}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 12
    fprintf('Testing Layer 12 (convolve)... ');
    current_output = apply_convolve(current_output, filterbanks{12}, biasvectors{12});
    if max(abs(current_output - layerResults{12}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 13
    fprintf('Testing Layer 13 (relu)... ');
    current_output = apply_relu(current_output);
    if max(abs(current_output - layerResults{13}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 14
    fprintf('Testing Layer 14 (convolve)... ');
    current_output = apply_convolve(current_output, filterbanks{14}, biasvectors{14});
    if max(abs(current_output - layerResults{14}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 15
    fprintf('Testing Layer 15 (relu)... ');
    current_output = apply_relu(current_output);
    if max(abs(current_output - layerResults{15}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 16
    fprintf('Testing Layer 16 (maxpool)... ');
    current_output = apply_maxpool(current_output);
    if max(abs(current_output - layerResults{16}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 17
    fprintf('Testing Layer 17 (fullconnect)... ');
    current_output = apply_fullconnect(current_output, filterbanks{17}, biasvectors{17});
    if max(abs(current_output - layerResults{17}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end
    % Layer 18
    fprintf('Testing Layer 18 (softmax)... ');
    current_output = apply_softmax(current_output);
    if max(abs(current_output - layerResults{18}), [], 'all') < 1e-6; fprintf('SUCCESS!\n'); else; fprintf('FAILURE.\n'); all_tests_passed = false; end

catch ME
    fprintf('-> TEST HALTED DUE TO ERROR: %s\n', ME.message);
    all_tests_passed = false;
end

% --- 3. Final Summary ---
if all_tests_passed
    fprintf('\n--- ALL 18 LAYERS PASSED THE INTEGRATION TEST! ---\n');
else
    fprintf('\n--- ONE OR MORE LAYERS FAILED. PLEASE REVIEW THE OUTPUT ABOVE. ---\n');
end
