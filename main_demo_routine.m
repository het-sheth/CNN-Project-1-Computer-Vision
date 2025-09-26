% This script serves as the main demonstration for the project. 
% It can be invoked with no arguments and will automatically run the test image from
% debuggingTest.mat through the full 18-layer CNN, displaying key
% intermediate and final results to demonstrate that the CNN is working
% as intended.

clear;
fprintf(' Running Project 1 Main Demo \n\n');

% Load All Necessary Data
fprintf('Loading data files\n');
try
    load 'debuggingTest.mat';
    load 'CNNparameters.mat';
    load 'cifar10testdata.mat';
catch
    error('FAILED: Could not find data files. Make sure they are in the same folder.');
end

% Run the Full 18-Layer Pipeline on the Test Image
fprintf('Running the test image through the full 18-layer CNN\n');

% We will store the output of Layer 2,8,11,17 for later visualization.
output_layer_2 = [];
output_layer_8 = [];
output_layer_11 = [];
output_layer_17 = [];

% Start with the raw input image
current_data = imrgb;

% Loop through all 18 layers
for d = 1:18
    layer_type = layertypes{d};
    
    % Call the appropriate function for the current layer
    switch layer_type
        case 'imnormalize'
            current_data = apply_imnormalize(current_data);
        case 'convolve'
            filters = filterbanks{d};
            biases = biasvectors{d};
            current_data = apply_convolve(current_data, filters, biases);
        case 'relu'
            current_data = apply_relu(current_data);
        case 'maxpool'
            current_data = apply_maxpool(current_data);
        case 'fullconnect'
            filters = filterbanks{d};
            biases = biasvectors{d};
            current_data = apply_fullconnect(current_data, filters, biases);
        case 'softmax'
            current_data = apply_softmax(current_data);
    end
    
    % Save the output of different layers when we pass it
    if d == 2
        output_layer_2 = current_data;
    elseif d == 8
        output_layer_8 = current_data;
    elseif d == 11
        output_layer_11 = current_data;
    elseif d == 17
        output_layer_17 = current_data;
    end
end

final_probabilities = current_data;
fprintf('Processing complete.\n\n');

fprintf('Generating figures for specified layers\n');

% Figure for Layer 2 (First Convolution)
figure('Name', 'Layer 2 Output (Convolution)');
sgtitle('Output Feature Maps from Layer 2');
num_channels_2 = size(output_layer_2, 3);
for i = 1:num_channels_2
    subplot(2, 5, i);
    feature_map = output_layer_2(:,:,i);
    imagesc(feature_map);
    colormap(gray);
    axis off;
    title(sprintf('Channel %d', i));
end

% Figure for Layer 8 (ReLU after Convolution)
figure('Name', 'Layer 8 Output (ReLU)');
sgtitle('Output Feature Maps from Layer 8');
num_channels_8 = size(output_layer_8, 3);
grid_size_8 = ceil(sqrt(num_channels_8));
for i = 1:num_channels_8
    subplot(grid_size_8, grid_size_8, i);
    feature_map = output_layer_8(:,:,i);
    imagesc(feature_map);
    colormap(gray);
    axis off;
    title(sprintf('Channel %d', i));
end

% Figure for Layer 11 (Maxpool)
figure('Name', 'Layer 11 Output (Maxpool)');
sgtitle('Output Feature Maps from Layer 11');
num_channels_11 = size(output_layer_11, 3);
grid_size_11 = ceil(sqrt(num_channels_11));
for i = 1:num_channels_11
    subplot(grid_size_11, grid_size_11, i);
    feature_map = output_layer_11(:,:,i);
    imagesc(feature_map);
    colormap(gray);
    axis off;
    title(sprintf('Channel %d', i));
end

% Figure for Layer 17 (Fully Connected)
figure('Name', 'Layer 17 Output (Fully Connected)');
bar(squeeze(output_layer_17));
title('Output Scores from Layer 17 (Before Softmax)');
xticklabels(classlabels);
xtickangle(45);
ylabel('Score');
grid on;

% Display the Original Image and Final Classification
figure('Name', 'Final Classification Result');
subplot(1, 2, 1); % Create a plot with two side-by-side sections
imagesc(imrgb);
axis off;
title('Original Input Image');

% Find the predicted class and its probability
[max_prob, predicted_class_index] = max(final_probabilities);
predicted_class_label = classlabels{predicted_class_index};

% Display the final bar chart
subplot(1, 2, 2);
bar(squeeze(final_probabilities));
title(sprintf('Final Prediction: %s (%.2f%% confidence)', predicted_class_label, max_prob*100));
xticklabels(classlabels);
xtickangle(45);
ylabel('Probability');
grid on;

fprintf('Displayed final classification result in a new figure window.\n');

fprintf('Demo complete.\n');
