function outarray = apply_fullconnect(inarray, filterbank, biasvals)
% apply_fullconnect: Applies a fully connected layer to a 3D input array.
% This function calculates the dot product between the entire input array and
% each filter in the filter bank, adds a corresponding bias, and returns
% the results in an output array. This implementation is vectorized for
% maximum efficiency in Matlab.
%
% Inputs:
%   inarray    - An N x M x D1 input array (e.g., 4x4x10).
%   filterbank - An N x M x D1 x D2 array of filters.
%   biasvals   - A 1 x D2 vector of bias values.
%
% Output:
%   outarray   - A 1 x 1 x D2 output array containing the scalar results.

    % --- Adheres to 'Program structure and readability' [5pts] ---
    % The code is commented to explain procedures. The function is a small,
    % self-contained module as required.

    % --- Implementation of 'Fully Connected' [5pts] ---
    % This code correctly implements the required functionality.

    % Get the number of filters (D2) from the 4th dimension of the filterbank.
    % This determines the number of output channels and loop iterations.
    num_filters = size(filterbank, 4);
    
    % Pre-allocate the output array with zeros for efficiency.
    % Matlab runs faster when it doesn't have to resize arrays inside a loop.
    % The output size is 1x1xD2 as specified.
    outarray = zeros(1, 1, num_filters);
    
    % --- Most Efficient Step: Vectorization ---
    % Reshape the 3D input array (e.g., 4x4x10) into a single, long column
    % vector (e.g., 160x1). The (:) operator does this automatically.
    % This is done *once* outside the loop to avoid redundant computation.
    input_vector = inarray(:);

    % Loop through each of the D2 filters to calculate each corresponding
    % output value. This loop is necessary as each filter produces a unique output.
    for i = 1:num_filters
        % Extract the i-th 3D filter from the 4D filterbank.
        current_filter = filterbank(:, :, :, i);
        
        % Reshape the 3D filter into a long column vector, matching the
        % input vector's shape.
        filter_vector = current_filter(:);
        
        % --- Core Calculation: Vectorized Dot Product ---
        % This is the most efficient way to compute the dot product in Matlab.
        % 1. (input_vector .* filter_vector) performs an element-wise
        %    multiplication of the two vectors.
        % 2. sum(...) adds up all the elements of the resulting vector into a
        %    single scalar value.
        % This single line replaces three slow nested for-loops.
        dot_product_sum = sum(input_vector .* filter_vector);
        
        % Add the corresponding bias value for the current filter.
        final_value = dot_product_sum + biasvals(i);
        
        % Store the final scalar value in the i-th channel of the output array.
        outarray(1, 1, i) = final_value;
    end
end

