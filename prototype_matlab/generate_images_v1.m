
%% generate images for Synthetic_v1
function [X, Y] = generate_images_v1( pos_num, neg_num, folder_name)
points_num = 20;
N = pos_num + neg_num;
X = zeros(N,points_num);
Y = zeros(N,points_num);

image = ones(288, 288, 3);
for i = 1:pos_num
    imshow(image);
    hold on;
    x = randi([10, 278], 1, points_num); 
    y = randi([10, 278], 1, points_num);
    X(i,:) = x;
    Y(i,:) = y;
    plot(x,y,'r.','MarkerSize',20);
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
    x = randi([10,278], 1, points_num); 
    y = randi([10,278], 1, points_num);
 
    %plot(x,y,'color.',rand(1,3),'MarkerSize',20);
    c = rand(1,3);
    if c ~= [1,0,0]
        X(i,:) = x;
        Y(i,:) = y;
        scatter(x,y,50,c,'filled');
        set(gca,'visible','off');
        s = sprintf('/%d', i);
        filename = strcat(folder_name, s);
        fig = gcf;
        fig.PaperUnits = 'inches';
        fig.PaperPosition = [0 0 2 2];
        print(gcf,filename,'-dpng','-r0');
        %print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');
    end
    hold off;
end

