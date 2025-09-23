% run_custom_image_test.m
% This script loads a custom image, prepares it for the CNN,
% runs it through the network, and analyzes the output.

clear;
clc;

% --- 1. Load the CNN Parameters and Labels ---
load 'CNNparameters.mat';
load 'cifar10testdata.mat'; % This file contains the 'classlabels'

% --- 2. Load and Prepare Your Custom Image ---
% Find an image on the web (e.g., a cat, a car, or your own face).
% Save it in your project folder.
image_filename = 'blank cat.png'; % <-- CHANGE THIS to your image file

% Read the image into MATLAB
full_size_image = imread(image_filename);

% The CNN requires a 32x32x3 input. We must resize the image.
% As discussed in Lecture 4, imresize correctly handles smoothing
% (anti-aliasing) to prevent artifacts.
input_image = imresize(full_size_image, [32, 32]);

% --- 3. Run the Image Through the CNN ---
% Call the "engine" function you created in Step 1.
final_probs = run_cnn_forward_pass(input_image, filterbanks, biasvectors);

% The output is a 1x1x10 array. Squeeze it into a simple vector.
final_probs = squeeze(final_probs);

% --- 4. Analyze and Display the Results ---
% Find the class with the highest probability.
[max_prob, predicted_class_index] = max(final_probs);
predicted_class_label = classlabels{predicted_class_index};

% Display the original image and the prediction.
figure;
subplot(1, 2, 1);
imshow(full_size_image);
title('Original Image');

subplot(1, 2, 2);
bar(final_probs);
title(sprintf('Prediction: %s (%.1f%%)', predicted_class_label, max_prob * 100));
set(gca, 'XTickLabel', classlabels, 'XTick', 1:10, 'XTickLabelRotation', 45);
ylabel('Probability');

% --- 5. Implement the "Unknown" Category Test ---
% This is the final part of 6e. We can create a simple rule: if the
% network is not very confident in its top choice, we can classify the
% object as "unknown". A good threshold is around 50-60%.

confidence_threshold = 0.5; % 50% confidence

if max_prob < confidence_threshold
    final_decision = 'Unknown';
else
    final_decision = predicted_class_label;
end

fprintf('Final Decision (with unknown category): %s\n', final_decision);
fprintf('Confidence in top choice: %.2f%%\n', max_prob * 100);
