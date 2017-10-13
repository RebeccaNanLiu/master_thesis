%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% manually generate images and extract HOG features
%% by Nan Liu
%% May. 08, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 0
%% generate 90 pos, 10 neg images for v6
folder_name = '/Users/nanliu/Documents/MATLAB/codeBooks/generated_v2';
[X, Y] = generate_images_v2( 90, 10, folder_name);
save X; 
save Y;
end
%% extract local HoG feature
load X;
load Y;
f = fopen('generated_v2.txt');  
positives = 90;
negatives = 10;
g = textscan(f,'%s','delimiter','\n');
fclose(f);
g = g{1};
N = size(g,1);
data = [];
cellSize = 8 ;
for i = 1 : N
    str = [g{i,:}];
    C = strsplit(str);
    str1 = strcat(C(:,1),'.png');
    img(i).path = str1{1};
    I = imread(img(i).path);
    
    I_padded = padarray(I,[40,40]);
    desc= [];
    for j = 1:10
        xmin = X(i, j);
        ymin = Y(i, j);
        I2 = imcrop(I_padded,[xmin ymin 80 80]);
        hog = vl_hog(im2single(I2), cellSize, 'verbose') ;
        sz = size(hog,1)*size(hog,2)*size(hog,3);
        hog_re = reshape(hog, sz, 1);
        desc = [desc hog_re];
    end
    img(i).descriptors = desc;
    data = [data desc];
    
end

%% PCA
[V, U] = pca(data);
figure; plot(sqrt(sum(U.^2, 1)));
data = U(:, 1:6)*V(:, 1:6)';

%% K-means
numCluster = 4;
[centers, assignments] = vl_kmeans(data, numCluster);

%% get the image indices for the descriptors, assignments//assignments_bro
assignments_bro = zeros(size(assignments), class(assignments));
for i = 1:100
    assignments_bro(1, 10*(i-1)+1:10*i) = i;
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
    for tao = 0: 0.01:1
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
figure;
plot(FPR(3,:), TPR(3,:), 'color','b','LineWidth',4);
axis([0 1 0 1]);
title('ROC');
xlabel('FPR');
ylabel('TPR');

if 0
%% draw ROC curve
figure;
plot(FPR(1,:), TPR(1,:), 'color','k');
hold on;
plot(FPR(2,:), TPR(2,:), 'color','g');
hold on;
plot(FPR(3,:), TPR(3,:), 'color','r');
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
title('ROC');
xlabel('FPR');
ylabel('TPR');
legend('5%','10%', '25%','50%', '60%', '70%', '80%', '90%');
end