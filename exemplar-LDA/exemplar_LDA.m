% author: Alexander Freytag
% date  : 27-02-2014 (dd-mm-yyyy)
% modified by Nan Liu, 03-2017


%% Training
% Load data 
f = fopen('bndbox_bottle.txt');             
g = textscan(f,'%s','delimiter','\n');
fclose(f);
g = g{1};

for i = 1 : size(g,1)
    str = [g{i,:}];
    C = strsplit(str);
    str1 = strcat(C(:,1),'.jpg');
    pos(i).im = str1{1};
    arr = cellfun(@str2num, C(:,2:end));
    pos(i).x1 = arr(1);
    pos(i).y1 = arr(2);
    pos(i).x2 = arr(3);
    pos(i).y2 = arr(4);
end

% We will train a model from a single instance.
% So let's create a tiny list of positive examples
%read image ...
N = size(g,1);

%for i = 1 : N
%    im = readImage(pos(i).im);
%    my_im_cell{i} = im;
    %figTrain = figure;
    %set ( figTrain, 'name', 'Training Image');
    %showboxes(im ,[pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2]);
%end

%settings for feature extraction
settings.i_binSize = 8;
settings.interval  = 10; % same as on who demo file
settings.order     = 20;
% note:
% change the representation as you desire. This repo comes along with HOG
% features as default, however, any diffferent feature type can be plugged
% in as well given the proper wrapper funtion.
% Examples can be found in our repository about patch discovery
settings.fh_featureExtractor = ...
  struct('name','Compute HOG features using WHO code', ...
         'mfunction',@computeHOGs_WHOorig, ...
         'b_leaveBoundary' , true );


% try locate previously trained bg-struct
try
    fileToBG = fullfile(pwd, 'data/bg11.mat');
    load( fileToBG );
    % compatibility to older versions
    if ( isfield(bg,'sbin') && ~isfield(bg, 'i_binSize') )
        bg.i_binSize = bg.sbin;
    end
catch
    % if not possible, leave bg empty and compute it from scratch lateron
    bg=[];
end

% no negative examples, use 'universal' negative model
neg   = [];
hitRate = zeros(N,N);
for k = 1:N
   
    model = learn_dataset( pos(k), neg, bg, settings );
    if (model.thresh == 0)
        continue;
    end
    %show learned model, i.e., visualized HOG feature for positive and negative weights
    b_closeImg = false;
    %showWeightVectorHOG( model.w, b_closeImg )

    %% Testing
    for i = 1:N
        fprintf('train:test =  %d-%d\n', k, i);
        test(1).im = pos(i).im;

        %perform detection
        boxes=test_dataset( test, model, settings );

        %convert output from 1x1-cell to double array
        boxes = boxes{1};
        %only take highest response
        %[~,bestIdx] = max ( boxes(:,1) );
        %boxes = boxes(bestIdx,:);

        %save the detection performance
        if (isempty(boxes) == 0)
            hitRate(k,i) = 1;
        end
    
        %show detections
        %im      = readImage(test(1).im);
    %    im = my_im_cell{3};
    %    figTest = figure;
    %    set ( figTest, 'name', 'Test Image');
    %    showboxes(im, boxes);
    end
    
    if (k == N && i == N)
        save hitRate;
    end
end
