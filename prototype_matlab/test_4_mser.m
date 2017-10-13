%% test for MSER

pfx = fullfile(vl_root,'data','spots.jpg') ;
I = imread(pfx) ;
image(I) ;

I = uint8(rgb2gray(I)) ;
regions = detectMSERFeatures(I);
[features, valid_points] = extractFeatures(I,regions,'Upright',true, 'SURFSize', 128);
figure; imshow(I); hold on;
plot(valid_points,'showOrientation',true);