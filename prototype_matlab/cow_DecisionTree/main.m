%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% train classifier to filter out images 
%% by Nan Liu
%% April. 03, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SIFT
f = fopen('train.txt');  
g = textscan(f,'%s','delimiter','\n');
fclose(f);
g = g{1};
N = size(g,1);
data = [];
for i = 1 : N
    str = [g{i,:}];
    C = strsplit(str);
    img(i).path = C{1};
    
    I = imread(img(i).path);
    I = single(rgb2gray(I)) ;
    % image(I);
    
    [f,d] = vl_sift(I) ;
    img(i).keypoints = f;
    img(i).descriptors = d;
    data = [data, d];
end
data = im2single(data);

%% k-means
numCluster = floor(sqrt(size(data,2)/2));
[centers, assignments] = vl_kmeans(data, numCluster);

%% build cluster structs
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

%% build X and Y
X = zeros(N, size(clusters,2));
for i = 1:size(clusters,2)
    C = unique(clusters(i).origionImgs);
    for j = 1:size(C,2)
        X(C(j), i)=1;
    end
end

for i = 1:151
    Y(i,:) = 1;
end
for i = 152:181
    Y(i,:) = 2;
end

%% decision classification tree
classifier = fitctree(X,Y);

%% making tests
f = fopen('test.txt');  
g = textscan(f,'%s','delimiter','\n');
fclose(f);
g = g{1};
N = size(g,1);

Xnew = zeros(N, numCluster);
for i = 1 : N
    str = [g{i,:}];
    C = strsplit(str);
    
    I = imread(C{1});
    I = single(rgb2gray(I)) ;
    
    [f,d] = vl_sift(I) ;
    IDX = knnsearch(centers',d');
    uniq_IDX = unique(IDX);

    for j = 1:size(uniq_IDX,1)
        Xnew(i, uniq_IDX(j)) = 1;
    end
end


Ynew = predict(classifier,Xnew);




