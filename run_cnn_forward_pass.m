function class_probabilities = run_cnn_forward_pass(image, filterbanks, biasvectors)
% run_cnn_forward_pass: Takes a 32x32x3 image and runs it through the full
% 18-layer CNN, returning the final 10-class probability vector.

    % Layer 1: Image Normalization
    current_output = apply_imnormalize(image);

    % Layer 2: Convolution
    current_output = apply_convolve(current_output, filterbanks{2}, biasvectors{2});

    % Layer 3: ReLU
    current_output = apply_relu(current_output);

    % Layer 4: Convolution
    current_output = apply_convolve(current_output, filterbanks{4}, biasvectors{4});

    % Layer 5: ReLU
    current_output = apply_relu(current_output);

    % Layer 6: Maxpool
    current_output = apply_maxpool(current_output);

    % Layer 7: Convolution
    current_output = apply_convolve(current_output, filterbanks{7}, biasvectors{7});
    
    % Layer 8: ReLU
    current_output = apply_relu(current_output);

    % Layer 9: Convolution
    current_output = apply_convolve(current_output, filterbanks{9}, biasvectors{9});

    % Layer 10: ReLU
    current_output = apply_relu(current_output);

    % Layer 11: Maxpool
    current_output = apply_maxpool(current_output);

    % Layer 12: Convolution
    current_output = apply_convolve(current_output, filterbanks{12}, biasvectors{12});

    % Layer 13: ReLU
    current_output = apply_relu(current_output);

    % Layer 14: Convolution
    current_output = apply_convolve(current_output, filterbanks{14}, biasvectors{14});

    % Layer 15: ReLU
    current_output = apply_relu(current_output);

    % Layer 16: Maxpool
    current_output = apply_maxpool(current_output);

    % Layer 17: Fully Connected
    current_output = apply_fullconnect(current_output, filterbanks{17}, biasvectors{17});

    % Layer 18: Softmax
    class_probabilities = apply_softmax(current_output);
end
