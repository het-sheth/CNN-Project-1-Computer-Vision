%%% Normalization Functionality
function outarray = apply_imnormalize(inarray)
    inarray_double = double(inarray);
    outarray = inarray_double / 255.0 - 0.5;
end

