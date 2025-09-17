%%% ReLU Functionality

function [out_red, out_green, out_blue] = reLU(im) 

    in_red = im(:,:,1); % parse for red channel of image
    in_green = im(:,:,2); % green channel
    in_blue = im(:,:,3); % blue channel
    
    out_red = max(in_red, 0);
    out_green = max(in_green, 0);
    out_blue = max(in_blue, 0);

end