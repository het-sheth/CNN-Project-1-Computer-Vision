% Creates an output array which replaces every negative number in
% the input array to 0
function outarray = apply_relu(inarray)
    outarray = max(inarray, 0);
end