function outarray = apply_fullconnect(inarray, filterbank, biasvals)
% This function calculates the dot product between the entire input array and
% each filter in the filter bank, adds a corresponding bias, and returns
% the results in an output array. This implementation is vectorized for
% maximum efficiency in Matlab.

    num_filters = size(filterbank, 4);
    
    % Pre-allocate the output array with zeros for efficiency.
    % Matlab runs faster when it doesn't have to resize arrays inside a loop.
    % The output size is 1x1xD2 as specified.
    outarray = zeros(1, 1, num_filters);
    
    % Reshape the 3D input array into a single, long column vector
    input_vector = inarray(:);

    % Loop through each of the D2 filters to calculate each corresponding
    % output value. This loop is necessary as each filter produces a unique output.
    for i = 1:num_filters
        current_filter = filterbank(:, :, :, i);
        
        filter_vector = current_filter(:);
        
        % This single line replaces three slow nested for-loops.
        dot_product_sum = sum(input_vector .* filter_vector);
        
        % Add the corresponding bias value for the current filter.
        final_value = dot_product_sum + biasvals(i);
        
        % Store the final scalar value in the i-th channel of the output array.
        outarray(1, 1, i) = final_value;
    end
end

