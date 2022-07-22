function I=get_luminance(Irgb)
    I=0.2126*Irgb(:,:,1)+0.7152*Irgb(:,:,2)+0.0722*Irgb(:,:,3);
end