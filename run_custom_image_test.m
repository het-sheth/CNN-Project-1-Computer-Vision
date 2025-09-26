% This script loads a custom image, prepares it for the CNN,
% runs it through the network, and analyzes the output.

clear;
clc;


load 'CNNparameters.mat';
load 'cifar10testdata.mat';

% Load and Prepare Custom Image
image_filename = 'Fred1.png';

full_size_image = imread(image_filename);

% The CNN requires a 32x32x3 input. We must resize the image.
input_image = imresize(full_size_image, [32, 32]);

% Run the Image Through the CNN
final_probs = run_cnn_forward_pass(input_image, filterbanks, biasvectors);

% Squeeze it into a simple vector.
final_probs = squeeze(final_probs);

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

% This is the final part of 6e. We can create a simple rule: if the
% network is not very confident in its top choice, we can classify
% it as unknown

confidence_threshold = 0.5; % 50% confidence

if max_prob < confidence_threshold
    final_decision = 'Unknown';
else
    final_decision = predicted_class_label;
end

fprintf('Final Decision (with unknown category): %s\n', final_decision);
fprintf('Confidence in top choice: %.2f%%\n', max_prob * 100);
