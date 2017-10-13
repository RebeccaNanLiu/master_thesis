%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simple baseline to filter out images 
%% by Nan Liu
%% Mar. 22, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LIFT
% read the images into img struct
f1 = fopen('banana.txt');  
g1 = textscan(f1,'%s','delimiter','\n');
fclose(f1);
g1 = g1{1};
for i = 1 : size(g1,1)
    str1 = [g1{i,:}];
    C1 = strsplit(str1);
    str11 = strcat(C1(:,1),'.png');
    img(i).path = str11{1};
end

% load the LIFT descriptors
f = fopen('banana_num.txt');  
positives = 65;
negatives = 5;
g = textscan(f,'%s','delimiter','\n');
fclose(f);
g = g{1};
N = size(g,1);
data = [];
for i = 1 : N
    str = [g{i,:}];
    C = strsplit(str);
    str1 = strcat('/Users/nanliu/Documents/MATLAB/codeBooks/banana_lift_desc/', C(:,1),'_desc.h5');
    d = h5read(str1{1},'/descriptors');
    img(i).descriptors = d;
    data = [data, d];
end


%% K-means
numCluster = 1024;
%numCluster = floor(sqrt(size(data,2)/2));
[centers, assignments] = vl_kmeans(data, numCluster);

%% get the image indices for the descriptors, assignments//assignments_bro
assignments_bro = zeros(size(assignments), class(assignments));
size_sum = 0;
size_cnt = 0;
for i = 1:size(img, 2)
    size_sum = size_sum + size(img(i).descriptors,2);

    if i==1
        assignments_bro(1,1:size_sum) = i;
    else 
        assignments_bro(1,size_cnt+1 : size_sum) = i; 
    end
    
    size_cnt = size_cnt + size(img(i).descriptors,2);
end

%% rank the clusters according to the cluster size (each image counted once)
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

%% visualize keypoints associated to "big" clusters 
if 0
% positives
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
    for i = CLUSTER:CLUSTER
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
% negatives
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
    for i = CLUSTER:CLUSTER 
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
            K_total = size(img(i).descriptors,2) ; 
    
            sel_pos = [];
            for j = 1:nclusters(1,nc) 
                imgInd = find(clusters(j).imageIndices==i);
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
        fp = size(pos_index,2)-border_flag;
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
title('ROC (CW=1024)');
xlabel('FPR');
ylabel('TPR');
legend('5%','10%', '25%','50%', '60%', '70%', '80%', '90%');
%saveas(gcf,'ROC.png');

