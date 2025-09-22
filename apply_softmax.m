% Tyler Cheng
function [outarray] = apply_softmax(inarray)
    alpha = max(inarray(:));
    stable_array = inarray - alpha;
    
    % Calculate the numerators (e^x for each element)
    numerators = exp(stable_array);
    
    % Calculate the denominator (the sum of all numerators)
    denominator = sum(numerators(:));
    
    % Divide to get the final probabilities.
    outarray = numerators / denominator;
end
