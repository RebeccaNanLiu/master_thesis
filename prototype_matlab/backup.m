%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% to back up 
%% by Nan Liu
%% Apr. 12, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SIFT
f = fopen('generated_v5.txt');  
positives = 90;
negatives = 10;
g = textscan(f,'%s','delimiter','\n');
fclose(f);
g = g{1};
N = size(g,1);
data = [];
feature = [];
for i = 1 : N
    str = [g{i,:}];
    C = strsplit(str);
    str1 = strcat(C(:,1),'.png');
    img(i).path = str1{1};
    I = imread(img(i).path);
    
    if 0
    figure;
    image(I);
    end;

    if (size(I, 3)>1)
        I = single(rgb2gray(I)) ;
    else 
        I = single(I);
    end
    
    peak_thresh = 0;
    edge_thresh = 10;
    [f,d] = vl_sift(I); %'PeakThresh', peak_thresh, 'edgethresh', edge_thresh) ;
    
    % visulize featurs 
    
    if 0
    perm = randperm(size(f,2)) ;
    sel = perm(1:size(f,2)) ;
    h1 = vl_plotframe(f(:,sel)) ;
    h2 = vl_plotframe(f(:,sel)) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','r','linewidth',2) ;
    end

    img(i).keypoints = f;
    img(i).descriptors = d;
    data = [data, d];
    feature = [feature, f];
end
data = im2single(data);
[ k ] = find_besk_k( data );

%% K-means
numCluster = floor(sqrt(size(data,2)/2));
[centers, assignments] = vl_kmeans(data, numCluster);


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
if 0
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
end

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

%% visualize keypoints associated to "big" clusters 
CLUSTER = 1;
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
    for i = CLUSTER:CLUSTER % 10 clusters
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
    for i = CLUSTER:CLUSTER % 10 clusters
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


%% Set the threshold
fid = fopen('result.txt', 'w');
fprintf(fid, 'TP\t FP\t FN\t TN\n');

counter = 0;
TP = [];
FP = [];
for i = 1:size(clusters,2)
    originImgs = [];
    counter = i;
    temp = 1;
    %%%%%%%%%%%%%%%%%%%%%%% OR between clusters %%%%%%%%%%%%%%%%%%%%%%%%%%
%    while(counter > 0)
%        originImgs = [originImgs, clusters(temp).imageIndices];
%        counter = counter - 1; 
%        temp = temp + 1;
%    end
    %%%%%%%%%%%%%%%%%%%%%%%% AND between clusters%%%%%%%%%%%%%%%%%%%%%%%%
    clusters_and = clusters(temp).imageIndices;
    for j = temp:counter-1
        clusters_and = intersect(clusters_and, clusters(temp+1).imageIndices);
        temp = temp + 1;
    end

    %fileind = unique(originImgs);
    fileind = unique(clusters_and);
    flag = find(ismember(fileind, positives));
    if (isempty(flag))
        flag = size(find(fileind < positives),2);
    end
    tp = flag;
    fp = size(fileind,2)-flag;
    tn = negatives - fp;
    fn = N-size(fileind,2)-tn;

    fprintf(fid, '%d  %d  %d  %d\n', tp, fp, fn, tn);
    TP = [TP, tp];
    FP = [FP, fp];
end

fclose(fid);

% draw ROC curve
TPR = TP/positives;
FPR = FP/negatives;

figure;
plot(FPR, TPR);
axis([0 1 0 1]);
title('STH ROC');
xlabel('FPR');
ylabel('TPR');
saveas(gcf,'ROC.png');

