% Compile mex functions
%
% feature extraction
mex -O features/featuresHOGGrayScale.cc -output features/featuresHOGGrayScale
mex -O features/featuresHOGColor.cc     -output features/featuresHOGColor
mex -O features/featuresHOGorig.cc      -output features/featuresHOGorig
%
% image resizing ( by scalar factors )
mex -O features/resizeGrayScale.cc      -output features/resizeGrayScale
mex -O features/resizeColor.cc          -output features/resizeColor
%
% image shrinking ( always by factor of 2 )
mex -O features/reduceGrayScale.cc      -output features/reduceGrayScale
mex -O features/reduceColor.cc          -output features/reduceColor
%
% discrete convolutions
mex -O features/fconv2D.cc              -output features/fconv2D
mex -O features/fconv3D.cc              -output features/fconv3D
