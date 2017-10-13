A = load('hitRate.mat');
B = A.hitRate;

if 0
% 1% noise 
C1 = B(1:677, 1:677);
D1 = cumsum(C1);
E1 = D1(677, :);
[F1,index] = sort(E1, 'descend');

fid = fopen('index.txt','wt');  % Note the 'wt' for writing in text mode
for i = 1:size(index,2)
    fprintf(fid,'%i\n',index(1,i)); 
end
fclose(fid);
end 

% 2% noise 
C2 = B(1:683, 1:683);
D2 = cumsum(C2);
E2 = D2(683, :);
[F2,index] = sort(E2, 'descend');

fid = fopen('index_2.txt','wt');  % Note the 'wt' for writing in text mode
for i = 1:size(index,2)
    fprintf(fid,'%i\n',index(1,i)); 
end
fclose(fid);

% 3% noise 
C3 = B(1:690, 1:690);
D3 = cumsum(C3);
E3 = D3(690, :);
[F3,index] = sort(E3, 'descend');

fid = fopen('index_3.txt','wt');  % Note the 'wt' for writing in text mode
for i = 1:size(index,2)
    fprintf(fid,'%i\n',index(1,i)); 
end
fclose(fid);

% 4% noise
C4 = B(1:697, 1:697);
D4 = cumsum(C4);
E4 = D4(697, :);
[F4,index] = sort(E4, 'descend');

fid = fopen('index_4.txt','wt');  % Note the 'wt' for writing in text mode
for i = 1:size(index,2)
    fprintf(fid,'%i\n',index(1,i)); 
end
fclose(fid);


% 5% noise 
C5 = B(1:704, 1:704);
D5 = cumsum(C5);
E5 = D5(704, :);
[F5,index] = sort(E5, 'descend');

fid = fopen('index_5.txt','wt');  % Note the 'wt' for writing in text mode
for i = 1:size(index,2)
    fprintf(fid,'%i\n',index(1,i)); 
end
fclose(fid);


% 10% noise
C10 = B(1:737, 1:737);
D10 = cumsum(C10);
E10 = D10(737, :);
[F10,index] = sort(E10, 'descend');

fid = fopen('index_10.txt','wt');  % Note the 'wt' for writing in text mode
for i = 1:size(index,2)
    fprintf(fid,'%i\n',index(1,i)); 
end
fclose(fid);


% 15% noise 
C15 = B(1:771, 1:771);
D15 = cumsum(C15);
E15 = D15(771, :);
[F15,index] = sort(E15, 'descend');

fid = fopen('index_15.txt','wt');  % Note the 'wt' for writing in text mode
for i = 1:size(index,2)
    fprintf(fid,'%i\n',index(1,i)); 
end
fclose(fid);


% 20% noise
C20 = B(1:804, 1:804);
D20 = cumsum(C20);
E20 = D20(804, :);
[F20,index] = sort(E20, 'descend');

fid = fopen('index_20.txt','wt');  % Note the 'wt' for writing in text mode
for i = 1:size(index,2)
    fprintf(fid,'%i\n',index(1,i)); 
end
fclose(fid);


