% Tyler Cheng
function [ outarray ] = apply_maxpool(inarray)
    % Top-left values of each 2x2 block
    A = inarray(1:2:end, 1:2:end, :);
    % Top-right values
    B = inarray(1:2:end, 2:2:end, :);
    % Bottom-left values
    C = inarray(2:2:end, 1:2:end, :);
    % Bottom-right values
    D = inarray(2:2:end, 2:2:end, :);
    
    % Find the element-wise maximum across the four arrays.
    % This effectively finds the max of each 2x2 block.
    outarray = max(max(A, B), max(C, D));
end
