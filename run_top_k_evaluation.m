clear;
clc;
close all;
fprintf('Running top-K classification');
try
    load 'CNNparameters.mat';
    load 'cifar10testdata.mat';
catch
    error('FAILED: Could not find data files. Make sure they are in the same folder.');
end

num_images = size(imageset, 4);
num_classes = length(classlabels);

top_k_correct_counts = zeros(1, num_classes);

for i = 1:num_images
    current_image = imageset(:,:,:,i);
    true_class_index = trueclass(i);
    
    final_probs = run_cnn_forward_pass(current_image, filterbanks, biasvectors);
    
    [~, sorted_indices] = sort(squeeze(final_probs), 'descend');
    
    for k = 1:num_classes
        top_k_predictions = sorted_indices(1:k);
        
        if ismember(true_class_index, top_k_predictions)
            top_k_correct_counts(k:end) = top_k_correct_counts(k:end) + 1;
            break;
        end
    end
end

top_k_accuracy = (top_k_correct_counts / num_images) * 100;
fprintf('Evaluation complete. Generating plot');
figure('Name', 'Top-K Classification Accuracy');
plot(1:num_classes, top_k_accuracy, '-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
title('Top-K Classification Accuracy');
xlabel('K (Number of Top Guesses Considered)');
ylabel('Accuracy (%)');
xticks(1:num_classes);
ylim([0 105]);

