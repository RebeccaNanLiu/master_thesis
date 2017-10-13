
%% generate images for Synthetic_v2
function [X, Y] = generate_images_v2( pos_num, neg_num, folder_name)
points_num = 10;
N = pos_num + neg_num;
X = zeros(N,points_num);
Y = zeros(N,points_num);

image = ones(288, 288, 3);
for i = 1:pos_num
    imshow(image);
    hold on;
    x = randi([20, 268], 1, points_num); 
    y = randi([20, 268], 1, points_num);
    X(i,:) = x;
    Y(i,:) = y;
    plot(x,y,'ro','MarkerSize',20);
    set(gca,'visible','off');
    set(gca,'position',[0 0 1 1],'units','normalized')
    s = sprintf('/%d', i);
    filename = strcat(folder_name, s);
    fig = gcf;
    fig.PaperUnits = 'inches';
    fig.PaperPosition = [0 0 2 2];
    print(gcf,filename,'-dpng','-r0');
    %print(gcf,filename,'-dpng','r0','-opengl');
    hold off;
end
for i = pos_num+1:N
    imshow(image);
    hold on;
    x = randi([30,258], 1, points_num); 
    y = randi([30,258], 1, points_num);
    X(i,:) = x;
    Y(i,:) = y;
    for j = 1:points_num
        rectangle('Position',[x(1,j) y(1,j) 25 25],'EdgeColor',[1,0,0]);
    end
    set(gca,'position',[0 0 1 1],'units','normalized')
    s = sprintf('/%d', i);
    filename = strcat(folder_name, s);
    fig = gcf;
    fig.PaperUnits = 'inches';
    fig.PaperPosition = [0 0 2 2];
    print(gcf,filename,'-dpng','-r0');
end