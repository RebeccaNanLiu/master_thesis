%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep feature based BOW
%% by Nan Liu
%% Apr. 28, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

positives = 90;
negatives = 10;
N = positives + negatives;
% read data
fileID = fopen('deep_car.txt','r');
formatSpec = '%f';
A = fscanf(fileID,formatSpec);
fclose(fileID);

data = [];
FEAT_SIZE = 2048;
for i = 1:N
    d = A(FEAT_SIZE*(i-1)+1:FEAT_SIZE*i, :);
    img(i).index = i;
    img(i).descriptors = d;
    data = [data d];
end 

%% K-means
numCluster = 4;
[centers, assignments] = vl_kmeans(data, numCluster, 'Initialization', 'plusplus');

%% get the image indices for the descriptors, assignments//assignments_bro
assignments_bro = zeros(size(assignments), class(assignments));
for i = 1:N
    assignments_bro(i) = i;
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
plot(FPR(3,:), TPR(3,:), 'color','k');
hold on;
plot(FPR(4,:), TPR(4,:), 'color','b');
hold on;
plot(FPR(5,:), TPR(5,:), 'color','y');
hold on;
plot(FPR(6,:), TPR(6,:), 'color','m');
hold on;
plot(FPR(7,:), TPR(7,:), 'color','c');
hold on;
plot(FPR(8,:), TPR(8,:),'--');
axis([0 1 0 1]);
title('ROC (CW=4)');
xlabel('FPR');
ylabel('TPR');
legend('5%','10%', '25%','50%', '60%', '70%', '80%', '90%');