%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simple baseline to filter out images 
%% generate (1 sample, 1 class) samples by (rot. + trans.+ scale.)
%% by Nan Liu
%% May. 09, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generate sample images 
t = 8;
for i = 1:90
    img = sprintf('%d.png', i);
    seed = imread(img);
    folder_name = '/Users/nanliu/Documents/MATLAB/codeBooks/generated_v_15/';
    %folder_name = '/Users/nanliu/Desktop/';
    %gen_num = [1,15,15,15,10,10,10,14];        % 1 sample
    %gen_num = [1,3,3,3,2,2,2,2];               % 5 samples 
    %gen_num = [1,1,1,1,1,1,1,2];               % 10 samples
    %gen_num = [1,1,0,0,1,1,1,1];               % 15 samples
    %gen_num = [1,0,0,0,0,0,0,2];               % 30 samples
    %gen_num = [1,0,0,0,0,0,0,1];               % 45 samples 
    %gen_num = [1,1,1,1,1,1,1,1];               % 45 samples 
    gen_num = [1,1,1,1,1,1,1,1];               % 90 samples
    generate_images_v3( seed, gen_num, folder_name, (i-1)*t);
end


if 0
%% SIFT  // MSER
f = fopen('generated_v_7.txt');  
positives = 90;
negatives = 10;
g = textscan(f,'%s','delimiter','\n');
fclose(f);
g = g{1};
N = size(g,1);
data = [];
feature = [];
des_num = [];

for i = 1 : N
    str = [g{i,:}];
    C = strsplit(str);
    str1 = strcat(C(:,1),'.jpeg');
    img(i).path = str1{1};
    I = imread(img(i).path);
    %disp(size(I))
    I = imresize(I, [224, 224]);
    %figure(); 
    %image(I);
    if (size(I, 3)>1)
        I_sift = single(rgb2gray(I)) ;
    else 
        I_sift = single(I);
    end
   
    peak_thresh = 0;
    edge_thresh = 10;
    [f,d] = vl_sift(I_sift, 'PeakThresh', peak_thresh, 'edgethresh', edge_thresh) ;
    
    if 0
    perm = randperm(size(f,2)) ;
    sel = perm ;
    h1 = vl_plotframe(f(:,sel)) ;
    h2 = vl_plotframe(f(:,sel)) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','y','linewidth',2) ;
  
    
    % MSER keypoints detection, SURF keypoints descriptor
    I_mser = uint8(rgb2gray(I)) ;
    regions = detectMSERFeatures(I_mser);
    [features, valid_points] = extractFeatures(I_mser,regions,'Upright',true, 'SURFSize', 128);
    f = transpose(features);
    d = f;
    end 
    
    img(i).keypoints = f;
    img(i).descriptors = d;
    data = [data, d];
    feature = [feature, f];
    des_num = [des_num, size(f,2)];
    
end

%save('des_num.mat', 'des_num');
data = im2single(data); 
%save('data.mat','data')

%% K-means
% 1024, 2048, 4096, 8192,16384, 32768, 65536
numCluster = 128;
%numCluster = floor(sqrt(size(data,2)/2));
[centers, assignments] = vl_kmeans(data, numCluster);
%% get the image indices for the descriptors, assignments//assignments_bro
assignments_bro = zeros(size(assignments), class(assignments));
size_sum = 0;
size_cnt = 0;
for i = 1:size(img, 2)
    size_cnt = size(img(i).descriptors,2);

    if i==1
        assignments_bro(1,1:size_cnt) = i;
    else 
        assignments_bro(1,size_sum+1 : size_sum+size_cnt) = i; 
    end
    
    size_sum = size_sum + size(img(i).descriptors,2);
end


if 0
%% rank the clusters according to the cluster size (each image counted once)
for i = 1: size(centers, 2)
    clusters(i).center = centers(:, i);
    clusters(i).descriptorIndices = find(assignments== i);
    % find the image indices that the clusters belong to
    for j = 1: size(clusters(i).descriptorIndices, 2)
        img_sum = 0;
        img_cnt = 0;
        for t = 1: size(img,2)
            img_sum = img_sum + size(img(t).descriptors, 2);
            img_cnt = img_cnt + 1;
            if clusters(i).descriptorIndices(:,j) <= img_sum
                clusters(i).imageIndices(:,j) = img_cnt;
                break;
            else 
                continue;
            end
        end
        
    end
end
%histogram_clusters = zeros(numCluster, 1);
for i = 1:size(clusters,2)
    C = unique(clusters(i).imageIndices);
    clusters(i).uniqueImgs = size(C,2);
    %histogram_clusters(i, 1) = size(C,2);
end
%histogram(histogram_clusters);
end 

%% rank the clusters according to the keypoints number
for i = 1: size(centers, 2)
    clusters(i).center = centers(:, i);
    clusters(i).descriptorIndices = find(assignments== i);
    % find the image indices that the clusters belong to
    for j = 1: size(clusters(i).descriptorIndices, 2)
        clusters(i).imageIndices(:,j) = assignments_bro(1, clusters(i).descriptorIndices(:,j)); 
    end
end

for i = 1:size(clusters,2)
    clusters(i).keyptanum = size(clusters(i).descriptorIndices, 2);
end

cells = struct2cell(clusters); %converts struct to cell matrix
sortvals = cells(4,1,:); % gets the values of the fourth field
mat = cell2mat(sortvals); % converts values to a matrix
mat = squeeze(mat); %removes the empty dimensions for a single vector
[sorted,ix] = sort(mat, 'descend'); %sorts the vector of values % descend, ascend
clusters = clusters(ix); %rearranges the original array

if 0
%% visualize keypoints associated to "big" clusters 
CLUSTER = 10;
figure;
for j = 1:10
    I = imread(img(j).path);
    subplot(3,5,j); 
    image(I);
        
    % plot all the features
    perm = randperm(size(img(j).keypoints,2)) ;
    sel_neg = perm(1:size(img(j).keypoints,2)) ; 
    h1 = vl_plotframe(img(j).keypoints(:,sel_neg)) ;
    h2 = vl_plotframe(img(j).keypoints(:,sel_neg)) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','r','linewidth',2) ;
     
    sel_pos = [];
    for i = 1:CLUSTER
        imgInd = find(clusters(i).imageIndices==j);
        for k = 1:size(imgInd, 2) 
            sel_pos = [sel_pos, clusters(i).descriptorIndices(imgInd(k))];
        end
        
        h3 = vl_plotframe(feature(:,sel_pos)) ;
        h4 = vl_plotframe(feature(:,sel_pos)) ;
        set(h3,'color','k','linewidth',3) ;
        set(h4,'color','g','linewidth',2) ;   
    end   
end

cnt = 10;
for j = 96:100
    cnt = cnt+1;
    I = imread(img(j).path);
    subplot(3,5,cnt); 
    image(I);
        
    % plot all the features
    perm = randperm(size(img(j).keypoints,2)) ;
    sel_neg = perm(1:size(img(j).keypoints,2)) ; 
    h1 = vl_plotframe(img(j).keypoints(:,sel_neg)) ;
    h2 = vl_plotframe(img(j).keypoints(:,sel_neg)) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','r','linewidth',2) ;
     
    sel_pos = [];
    for i = 1:CLUSTER 
        imgInd = find(clusters(i).imageIndices==j);
        for k = 1:size(imgInd, 2) 
            sel_pos = [sel_pos, clusters(i).descriptorIndices(imgInd(k))];
        end
        
        h3 = vl_plotframe(feature(:,sel_pos)) ;
        h4 = vl_plotframe(feature(:,sel_pos)) ;
        set(h3,'color','k','linewidth',3) ;
        set(h4,'color','g','linewidth',2) ;   
    end   
end
end

%% Set the threshold
nclusters = [ceil(size(clusters,2) * 0.05), ceil(size(clusters,2) * 0.1)...
    ceil(size(clusters,2) * 0.25), ceil(size(clusters,2) * 0.5)...
    ceil(size(clusters,2) * 0.6), ceil(size(clusters,2) * 0.7)...
    ceil(size(clusters,2) * 0.8), ceil(size(clusters,2) * 0.9)];
for nc = 1:size(nclusters, 2)
    TP = [];
    FP = [];
    for tao = 0: 0.02:1
        % compare K_inlier/K_total with tao
        flag = zeros(1,N);
    
        for i = 1:N
            K_total = size(img(i).keypoints,2) ; 
    
            sel_pos = [];
            for j = 1:nclusters(1,nc) 
                imgInd = find(clusters(j).imageIndices==i);%local
                for k = 1:size(imgInd, 2) 
                    sel_pos = [sel_pos, clusters(j).descriptorIndices(imgInd(k))];
                end    
            end
    
            K_inlier = size(sel_pos, 2);
    
            t = double(K_inlier/K_total);
            if t >= tao
                flag(1,i) = 1;
            end  
        end 
    
        % compute TPR/FPR
        pos_index = find(flag == 1);
        border_flag = find(ismember(pos_index, positives));
        if (isempty(border_flag))
            border_flag = size(find(pos_index < positives),2);
        end
        tp = border_flag;
        fp_count = 0;
        for i = 91:100
            if flag(1,i) == 1
                fp_count = fp_count + 1;
            end
        end
        fp = fp_count;
        %fp = size(pos_index,2)-border_flag; % it is right version 
        tn = negatives - fp;
        fn = N-size(pos_index,2)-tn;

        TP = [TP, tp];
        FP = [FP, fp];
    end
    tpr = TP/positives;
    fpr = FP/negatives;
    
    TPR(nc,:) = tpr;
    FPR(nc,:) = fpr;
end


%% draw ROC curve
figure;
plot(FPR(1,:), TPR(1,:), 'color','r');
hold on;
plot(FPR(2,:), TPR(2,:), 'color','g');
hold on;
plot(FPR(3,:), TPR(3,:), 'color','b');
hold on;
plot(FPR(4,:), TPR(4,:), 'color','k');
hold on;
plot(FPR(5,:), TPR(5,:), 'color','y');
hold on;
plot(FPR(6,:), TPR(6,:), 'color','m');
hold on;
plot(FPR(7,:), TPR(7,:), 'color','c');
hold on;
plot(FPR(8,:), TPR(8,:),'--');
axis([0 1 0 1]);
title('ROC');
xlabel('FPR');
ylabel('TPR');
legend('5%','10%', '25%','50%', '60%', '70%', '80%', '90%');
%saveas(gcf,'ROC.png');
end