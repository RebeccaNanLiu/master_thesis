%% generate images for Synthetic_v3
function generate_images_v3( seed, gen_num, folder_name, N)
%% generate samples for v7,v8,v9, ...
% rotation range = [-10, 10]
% translation range = [-40, 40]
% scale range = [0.1, 4]
% seed image(s)
for i = 1:sum(gen_num(1:1))
    s = sprintf('/%d.jpeg', i+N);
    filename = strcat(folder_name, s);
    imwrite(seed, filename);
    %figure; imshow(seed);
end 

% rotation
cnt = 1;
rot = [randperm(10,10) -randperm(10,5)];
for i = sum(gen_num(1:1))+1:sum(gen_num(1:2))
    s = sprintf('/%d.jpeg', i+N);
    filename = strcat(folder_name, s);
        
    r = imrotate255(seed(:,:,1),rot(cnt), 'crop');
    g = imrotate255(seed(:,:,2),rot(cnt), 'crop');
    b = imrotate255(seed(:,:,3),rot(cnt), 'crop');
    I = cat(3, r, g, b);
    imwrite(I, filename);
    %figure;imshow(I);
    cnt = cnt +1;
end

% translation
cnt = 1;
trans_x = [randperm(30,10) -randperm(30,5)];
trans_y = [-randperm(30,10) randperm(30,5)];
for i = sum(gen_num(1:2))+1:sum(gen_num(1:3))
    s = sprintf('/%d.jpeg', i+N);
    filename = strcat(folder_name, s);
    
    r = imtranslate(seed(:,:,1), [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    g = imtranslate(seed(:,:,2), [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    b = imtranslate(seed(:,:,3), [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    I = cat(3, r, g, b);
    imwrite(I, filename);
    %figure;imshow(I);
    cnt = cnt + 1;
end

% scale
cnt = 1;
scale = randperm(20, 15)/10;
for i = sum(gen_num(1:3))+1:sum(gen_num(1:4))
    s = sprintf('/%d.jpeg', i+N);
    filename = strcat(folder_name, s);
    if scale(cnt)==1
        scale(cnt)== 1.2;
    end
    r = imresize(seed(:,:,1), scale(cnt));
    g = imresize(seed(:,:,2), scale(cnt));
    b = imresize(seed(:,:,3), scale(cnt));
    I = cat(3, r, g, b);
    imwrite(I, filename);
    %figure;imshow(I);
    cnt = cnt + 1;
end

% rotation+translation
cnt = 1;
rot = [randperm(10, 5) -randperm(10, 5)];
trans_x = [randperm(15, 5) -randperm(15, 5)];
trans_y = [randperm(15,5) -randperm(15, 5)];
for i = sum(gen_num(1:4))+1:sum(gen_num(1:5))
    s = sprintf('/%d.jpeg', i+N);
    filename = strcat(folder_name, s);
    
    r = imtranslate(seed(:,:,1), [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    g = imtranslate(seed(:,:,2), [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    b = imtranslate(seed(:,:,3), [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    
    r = imrotate255(r, rot(cnt), 'crop');
    g = imrotate255(g, rot(cnt), 'crop');
    b = imrotate255(b, rot(cnt), 'crop');
    
    I = cat(3, r, g, b);
    imwrite(I, filename);
    %figure;imshow(I);
    cnt = cnt + 1;
end

% rotation + scale
cnt = 1;
rot = [randperm(10, 5) -randperm(10, 5)];
scale = randperm(20,15)/10;
for i = sum(gen_num(1:5))+1:sum(gen_num(1:6))
    s = sprintf('/%d.jpeg', i+N);
    filename = strcat(folder_name, s);
    
    r = imresize(seed(:,:,1), scale(cnt));
    g = imresize(seed(:,:,2), scale(cnt));
    b = imresize(seed(:,:,3), scale(cnt));
    
    r = imrotate255(r, rot(cnt), 'crop');
    g = imrotate255(g, rot(cnt), 'crop');
    b = imrotate255(b, rot(cnt), 'crop');
    
    I = cat(3, r, g, b);
    imwrite(I, filename);
    %figure;imshow(I);
    cnt = cnt + 1;
end

% translation + scale
cnt = 1;
trans_x = [randperm(15,5) -randperm(15,5)];
trans_y = [randperm(15,5) -randperm(15,5)];
scale = randperm(20,15)/10;
for i = sum(gen_num(1:6))+1:sum(gen_num(1:7))
    s = sprintf('/%d.jpeg', i+N);
    filename = strcat(folder_name, s);
    
    r = imresize(seed(:,:,1), scale(cnt));
    g = imresize(seed(:,:,2), scale(cnt));
    b = imresize(seed(:,:,3), scale(cnt));
    
    r = imtranslate(r, [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    g = imtranslate(g, [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    b = imtranslate(b, [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    
    I = cat(3, r, g, b);
    imwrite(I, filename);
    %figure;imshow(I);
    cnt = cnt + 1;
end

% rotation + translation + scale 
cnt = 1;
rot = [randperm(10, 7) -randperm(10, 7)];
trans_x = [randperm(15,7) -randperm(15,7)];
trans_y = [randperm(15,7) -randperm(15,7)];
scale = randperm(20,15)/10;
for i = sum(gen_num(1:7))+1:sum(gen_num(1:8))
    s = sprintf('/%d.jpeg', i+N);
    filename = strcat(folder_name, s);
    
    r = imresize(seed(:,:,1), scale(cnt));
    g = imresize(seed(:,:,2), scale(cnt));
    b = imresize(seed(:,:,3), scale(cnt));
    
    r = imtranslate(r, [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    g = imtranslate(g, [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    b = imtranslate(b, [trans_x(cnt) trans_y(cnt)], 'FillValues', 255);
    
    r = imrotate255(r, rot(cnt), 'crop');
    g = imrotate255(g, rot(cnt), 'crop');
    b = imrotate255(b, rot(cnt), 'crop');
    
    I = cat(3, r, g, b);
    imwrite(I, filename);
    %figure;imshow(I);
    cnt = cnt + 1;
end



if 0
%% for testing
%% generate images for v4
image=ones(64,64,3);
for i = 1:pos_num
    x = randi([1,64], 1, points_num); 
    y = randi([1,64], 1, points_num);
    plot(x,y,'r.','MarkerSize',160);
    set(gca,'visible','off')
    s = sprintf('/%d', i);
    filename = strcat(folder_name, s);
    print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');
end

for i = pos_num+1:N
    x = randi([1,64], 1, points_num); 
    y = randi([1,64], 1, points_num);
    %plot(x,y,'color.',rand(1,3),'MarkerSize',20);
    c = rand(1,3);
    scatter(x,y,25,c,'filled');
    set(gca,'visible','off')
    s = sprintf('/%d', i);
    filename = strcat(folder_name, s);
    print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');
end
%% for generated_v3
image=ones(64,64,3); 
coord_x = randi([24,40], 1, N); 
coord_y = randi([24,40], 1, N);
%r = randi([1,20], 1, 100);
r = 24;
for i = 1:pos_num
    figure; imshow(image);
    hold on;
    th = 0:pi/50:2*pi;
    x = r * cos(th) + coord_x(i);
    y = r * sin(th) + coord_y(i);
    plot(x, y);
    fill(x, y, [1,0,0]);
    
    filename = sprintf('/Users/nanliu/Documents/MATLAB/codeBooks/generated_v3/%d', i);
    print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');
    %saveas(gcf, filename, 'jpg'); 
    hold off;
end


pos_line = [30, 50, 40,41, 25, 40, 38, 16];
RGB = insertShape(image,'Line',{pos_line},'Color', {'green'},'Opacity',1);
figure;imshow(RGB);
hold on;
x = coord_x(91);
y = coord_y(91);
rectangle('Position',[25 25 20 45],'facecolor',[0,1,0]);
filename = sprintf('/Users/nanliu/Documents/MATLAB/codeBooks/generated_v3/91');
print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');
hold off;

pos_line = [10,10, 25, 35];
RGB = insertShape(image,'Line',{pos_line},'Color', {'green'},'Opacity',1);
figure;imshow(RGB);
hold on;
x = coord_x(92);
y = coord_y(92);
rectangle('Position',[30 30 25 27],'facecolor',[0,1,0]);
filename = sprintf('/Users/nanliu/Documents/MATLAB/codeBooks/generated_v3/92');
print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');
hold off;

figure;imshow(image);
hold on;
x = coord_x(93);
y = coord_y(93);
rectangle('Position',[25 20 20 15],'facecolor',[0,1,0]);
filename = sprintf('/Users/nanliu/Documents/MATLAB/codeBooks/generated_v3/93');
print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');
hold off;

figure;imshow(image);
hold on;
x = coord_x(94);
y = coord_y(94);
rectangle('Position',[x y 40 45],'facecolor',[0,1,0]);
filename = sprintf('/Users/nanliu/Documents/MATLAB/codeBooks/generated_v3/94');
print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');
hold off;

pos_triangle = [15 19 30 60 50 29];
RGB = insertShape(image,'FilledPolygon',{pos_triangle},'Color', {'green'},'Opacity',1);
figure;imshow(RGB);
filename = sprintf('/Users/nanliu/Documents/MATLAB/codeBooks/generated_v3/95');
print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');

pos_triangle = [18 29 30 1 60 29];
RGB = insertShape(image,'FilledPolygon',{pos_triangle},'Color', {'green'},'Opacity',1);
figure;imshow(RGB);
filename = sprintf('/Users/nanliu/Documents/MATLAB/codeBooks/generated_v3/96');
print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');

pos_triangle = [28 39 50 25 40 49];
RGB = insertShape(image,'FilledPolygon',{pos_triangle},'Color', {'green'},'Opacity',1);
figure;imshow(RGB);
filename = sprintf('/Users/nanliu/Documents/MATLAB/codeBooks/generated_v3/97');
print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');

pos_hexagon = [30 16 15 18 50 27 20 29 60 25 30 55];
RGB = insertShape(image,'FilledPolygon',{pos_hexagon},'Color', {'green'},'Opacity',1);
figure;imshow(RGB);
filename = sprintf('/Users/nanliu/Documents/MATLAB/codeBooks/generated_v3/98');
print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');

pos_hexagon = [23 13 30 18 30 45 33 25 36 59 23 53];
RGB = insertShape(image,'FilledPolygon',{pos_hexagon},'Color', {'green'},'Opacity',1);
figure;imshow(RGB);
filename = sprintf('/Users/nanliu/Documents/MATLAB/codeBooks/generated_v3/99');
print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');

pos_hexagon = [6 6 30 18 30 50 33 29 36 50 36 19];
RGB = insertShape(image,'FilledPolygon',{pos_hexagon},'Color', {'green'},'Opacity',1);
figure;imshow(RGB);
filename = sprintf('/Users/nanliu/Documents/MATLAB/codeBooks/generated_v3/100');
print(gcf,filename,'-dpng',['-r',num2str(64)],'-opengl');
end
end