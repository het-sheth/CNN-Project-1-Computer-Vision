function outarray = apply_convolve(inarray, filterbank, biasvals)

[N, M, D1] = size(inarray);
D2 = size(filterbank, 4);


outarray = zeros(N, M, D2);

for l = 1:D2
    

    summed_conv_result = zeros(N, M);
    
    for k = 1:D1
        
        input_channel = inarray(:,:,k);
        
        filter_channel = filterbank(:,:,k,l);
      
        conv_channel_result = imfilter(input_channel, filter_channel, 'conv', 0);
        
        summed_conv_result = summed_conv_result + conv_channel_result;
        
    end
    

    final_channel = summed_conv_result + biasvals(l);
    
    outarray(:,:,l) = final_channel;
    
end

end

