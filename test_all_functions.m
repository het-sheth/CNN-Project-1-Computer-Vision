% test_all_functions.m
% This script efficiently tests all required CNN layer functions against the
% known correct outputs provided in debuggingTest.mat.

clear;
clc;
fprintf('--- Starting Comprehensive Test of All CNN Layer Functions ---\n\n');

% --- Load Data Just Once ---
try
    load 'debuggingTest.mat';
    load 'CNNparameters.mat';
    fprintf('Successfully loaded testing and parameter files.\n\n');
catch
    error('FAILED: Could not find debuggingTest.mat or CNNparameters.mat. Make sure they are in the same folder.');
end

% --- Test Each Function Sequentially ---

% 1. Test apply_imnormalize (Layer 1)
fprintf('1. Testing apply_imnormalize...\n');
try
    my_output = apply_imnormalize(imrgb);
    correct_output = layerResults{1};
    if max(abs(my_output - correct_output), [], 'all') < 1e-6
        fprintf('   -> SUCCESS!\n\n');
    else
        fprintf('   -> FAILURE: Output does not match expected result.\n\n');
    end
catch e
    fprintf('   -> ERROR: The function failed to run. Message: %s\n\n', e.message);
end

% 2. Test apply_convolve (Layer 2)
fprintf('2. Testing apply_convolve...\n');
try
    my_output = apply_convolve(layerResults{1}, filterbanks{2}, biasvectors{2});
    correct_output = layerResults{2};
    if max(abs(my_output - correct_output), [], 'all') < 1e-6
        fprintf('   -> SUCCESS!\n\n');
    else
        fprintf('   -> FAILURE: Output does not match expected result.\n\n');
    end
catch e
    fprintf('   -> ERROR: The function failed to run. Message: %s\n\n', e.message);
end

% 3. Test apply_relu (Layer 3)
fprintf('3. Testing apply_relu...\n');
try
    my_output = apply_relu(layerResults{2});
    correct_output = layerResults{3};
    if max(abs(my_output - correct_output), [], 'all') < 1e-6
        fprintf('   -> SUCCESS!\n\n');
    else
        fprintf('   -> FAILURE: Output does not match expected result.\n\n');
    end
catch e
    fprintf('   -> ERROR: The function failed to run. Message: %s\n\n', e.message);
end

% 4. Test apply_maxpool (Layer 6)
fprintf('4. Testing apply_maxpool...\n');
try
    my_output = apply_maxpool(layerResults{5});
    correct_output = layerResults{6};
    if max(abs(my_output - correct_output), [], 'all') < 1e-6
        fprintf('   -> SUCCESS!\n\n');
    else
        fprintf('   -> FAILURE: Output does not match expected result.\n\n');
    end
catch e
    fprintf('   -> ERROR: The function failed to run. Message: %s\n\n', e.message);
end

% 5. Test apply_fullconnect (Layer 17)
fprintf('5. Testing apply_fullconnect...\n');
try
    my_output = apply_fullconnect(layerResults{16}, filterbanks{17}, biasvectors{17});
    correct_output = layerResults{17};
    if max(abs(my_output - correct_output), [], 'all') < 1e-6
        fprintf('   -> SUCCESS!\n\n');
    else
        fprintf('   -> FAILURE: Output does not match expected result.\n\n');
    end
catch e
    fprintf('   -> ERROR: The function failed to run. Message: %s\n\n', e.message);
end

% 6. Test apply_softmax (Layer 18)
fprintf('6. Testing apply_softmax...\n');
try
    my_output = apply_softmax(layerResults{17});
    correct_output = layerResults{18};
    if max(abs(my_output - correct_output), [], 'all') < 1e-6
        fprintf('   -> SUCCESS!\n\n');
    else
        fprintf('   -> FAILURE: Output does not match expected result.\n\n');
    end
catch e
    fprintf('   -> ERROR: The function failed to run. Message: %s\n\n', e.message);
end
