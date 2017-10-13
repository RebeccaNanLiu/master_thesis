function [ k ] = find_besk_k( data )
% find the best cluster number (k) for k-means
% using the Elbow method 

t = floor(sqrt(size(data,2)/2));
numClusters = [t, 100, 200, 300, 400, 500, 600, 700, 800, 900,1000];

sse_sum = zeros(1, size(numClusters, 2));
for k_num = 1:size(numClusters, 2)
    [centers, assignments] = vl_kmeans(data, numClusters(:,k_num));

    for i = 1:size(assignments,2)
        sse_sum(:, k_num) = sse_sum(:,k_num) + norm(centers(:, assignments(1,i)) - data(:,i));
    end
end

figure;
plot(numClusters,sse_sum,'--gs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])        

title('SSE-K')
xlabel('k')
ylabel('SSE')
saveas(gcf,'SSE_K.png')

% find the elbow point
for i = 2: size(numClusters,2)-1
    secondDerivative(:,i) = sse_sum(:,i+1) + sse_sum(:,i-1) - 2 * sse_sum(:,i);
end

[~,i] = max(secondDerivative(:)); % the point with the biggest curvature

k = i + 1;
end

