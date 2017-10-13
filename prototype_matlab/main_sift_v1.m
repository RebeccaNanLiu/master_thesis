%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simple baseline to filter out images 
%% by Nan Liu
%% April. 4, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SIFT
f = fopen('cow.txt');  
positives = 303;
negatives = 61;
g = textscan(f,'%s','delimiter','\n');
fclose(f);
g = g{1};
N = size(g,1);
data = [];
for i = 1 : N
    str = [g{i,:}];
    C = strsplit(str);
    str1 = strcat(C(:,1),'.jpg');
    img(i).path = str1{1};
    
    I = imread(img(i).path);
    I = single(rgb2gray(I)) ;
    % image(I);
    
    [f,d] = vl_sift(I) ;
    img(i).keypoints = f;
    img(i).descriptors = d;
    data = [data, d];
end
data = im2single(data);

%% K-means
numCluster = floor(sqrt(size(data,2)/2));
[centers, assignments] = vl_kmeans(data, numCluster);

%% build histogram matrix 
for i = 1: size(centers, 2)
    clusters(i).center = centers(:, i);
    clusters(i).neighbours = find(assignments== i);
    % find the image indices that the clusters belong to
    for j = 1: size(clusters(i).neighbours, 2)
        img_sum = 0;
        img_cnt = 0;
        for t = 1: size(img,2)
            img_sum = img_sum + size(img(t).descriptors, 2);
            img_cnt = img_cnt + 1;
            if clusters(i).neighbours(:,j) <= img_sum
                clusters(i).origionImgs(:,j) = img_cnt;
                break;
            else 
                continue;
            end
        end
    end
end

for i = 1:size(clusters,2)
    C = unique(clusters(i).origionImgs);
    clusters(i).uniqueImgs = size(C,2);
end



if 0
%% Set the threshold
fid = fopen('result_forward.txt', 'w');

counter = 0;
TP = [];
FP = [];
for i = 1:size(clusters,2)
    originImgs = [];
    counter = i;
    temp = 1;
    %%%%%%%%%%%%%%%%%%%%%%% OR between clusters %%%%%%%%%%%%%%%%%%%%%%%%%%
%    while(counter > 0)
%        originImgs = [originImgs, clusters(temp).origionImgs];
%        counter = counter - 1; 
%        temp = temp + 1;
%    end
    %%%%%%%%%%%%%%%%%%%%%%%% AND between clusters%%%%%%%%%%%%%%%%%%%%%%%%
    clusters_and = clusters(temp).origionImgs;
    for j = temp:counter-1
        clusters_and = intersect(clusters_and, clusters(temp+1).origionImgs);
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

    fprintf(fid, '%d %d %d %d\n', tp, fp, fn, tn);
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
title('COW ROC');
xlabel('FPR');
ylabel('TPR');
saveas(gcf,'ROC_forward.png');
end