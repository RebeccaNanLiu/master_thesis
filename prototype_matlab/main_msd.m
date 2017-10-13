%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simple baseline to filter out images 
%% generate (1 sample, 1 class) samples by (rot. + trans.+ scale.)
%% by Nan Liu
%% May. 30, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% msd+surf 
M = csvread('descriptors.csv');
desc = transpose(M);
desc_cnt = csvread('desc_cnt.csv');

f = fopen('generated_v13.txt');  
positives = 90;
negatives = 10;
g = textscan(f,'%s','delimiter','\n');
fclose(f);
g = g{1};
N = size(g,1);
cnt = 0;
sum = 0;
for i = 1 : N
    str = [g{i,:}];
    C = strsplit(str);
    str1 = strcat(C(:,1),'.png');
    img(i).path = str1{1};
    I = imread(img(i).path);
    
    cnt = desc_cnt(i);
    
    if i == 1
        img(i).descriptors = desc(:,1:cnt);
    else 
        img(i).descriptors = desc(:,sum+1:sum+cnt);
    end
    
    sum = sum + cnt;
    
end
data = im2single(desc); 

%% K-means
% 1024, 2048, 4096, 8192,16384, 32768? 65536
numCluster = 2048 ;
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
title('ROC');
xlabel('FPR');
ylabel('TPR');
legend('5%','10%', '25%','50%', '60%', '70%', '80%', '90%');
%saveas(gcf,'ROC.png');
 