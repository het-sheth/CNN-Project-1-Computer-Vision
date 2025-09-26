% This script runs the entire cifar10 test set through the trained CNN,
% calculates the confusion matrix, and computes the overall accuracy.

clear;
clc;

fprintf('Starting Performance Evaluation\n');
load 'CNNparameters.mat';
load 'cifar10testdata.mat'; 


num_images = size(imageset, 4);
num_classes = length(classlabels);
% This creates our empty 10x10 grid for the confusion matrix.
confusion_matrix = zeros(num_classes, num_classes);
correct_predictions = 0;

fprintf('Beginning classification of %d images\n', num_images);

% This loop will take some time to run.
for i = 1:num_images
    % Get the current image
    current_image = imageset(:,:,:,i);
    
    % Get the true class label for this image (a number from 1 to 10)
    true_class_index = trueclass(i);
    
    % Run the image through your complete CNN "engine" function
    final_probs = run_cnn_forward_pass(current_image, filterbanks, biasvectors);
    
    % Find the predicted class (the one with the highest probability)
    [~, predicted_class_index] = max(final_probs);
    
    % Update the Confusion Matrix and Accuracy Count
    confusion_matrix(true_class_index, predicted_class_index) = ...
        confusion_matrix(true_class_index, predicted_class_index) + 1;
        
    % Check if the prediction was correct
    if predicted_class_index == true_class_index
        correct_predictions = correct_predictions + 1;
    end
    
    % Display progress every 500 images
    if mod(i, 500) == 0
        fprintf('  ...processed %d / %d images.\n', i, num_images);
    end
end

fprintf('Classification complete.\n\n');

% Calculate and Display Final Results
overall_accuracy = correct_predictions / num_images;

fprintf('Performance Results\n');
fprintf('Overall Classification Accuracy: %.2f%%\n\n', overall_accuracy * 100);

fprintf('Confusion Matrix:\n');
fprintf('(Rows are True Class, Columns are Predicted Class)\n\n');
disp(confusion_matrix);