
fid = fopen('nan.txt', 'w');
for i = 1:864
    fprintf(fid, '/Users/nanliu/hypercolumns/generated_90_aug/%d\n', i);
end

fclose(fid);