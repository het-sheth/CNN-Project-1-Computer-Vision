%%% Normalization Functionality
function [out_red, out_green, out_blue] = apply_imnormalize(inarray) 

    in_red = inarray(:,:,1); % parse for red channel of image
    in_green = inarray(:,:,2); % green channel
    in_blue = inarray(:,:,3); % blue channel
    
    out_red = in_red/255.0 - 0.5;
    out_green = in_green/255.0 - 0.5;
    out_blue = in_blue/255.0 - 0.5;

end

