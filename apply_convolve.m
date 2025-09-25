function outarray = apply_convolve(inarray, filterbank, biasvals)

% Get input and output dimensions.
[N, M, D1] = size(inarray);
D2 = size(filterbank, 4);

% Initialize the output array with the correct dimensions.
outarray = zeros(N, M, D2);

% Iterate through each filter to compute one output channel at a time.
for l = 1:D2
    
    % Accumulator for the current output channel.
    summed_conv_result = zeros(N, M);
    
    % Convolve each input channel with the corresponding filter channel.
    for k = 1:D1
        
        input_channel = inarray(:,:,k);
        filter_channel = filterbank(:,:,k,l);
      
        % Perform 2D convolution with zero-padding.
        conv_channel_result = imfilter(input_channel, filter_channel, 'conv', 0);
        
        % Add the result to the accumulator.
        summed_conv_result = summed_conv_result + conv_channel_result;
        
    end
    
    % Add the scalar bias value to the summed convolutions.
    final_channel = summed_conv_result + biasvals(l);
    
    % Store the result in the corresponding output channel.
    outarray(:,:,l) = final_channel;
    
end

end

