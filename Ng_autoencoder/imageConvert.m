load IMAGES;

fileID = fopen('images.dat','w');
fwrite(fileID, 100, 'int32');
fwrite(fileID, 10, 'int32');
fwrite(fileID, 512, 'int32');
fwrite(fileID, 512, 'int32');

for i=1:10
    fwrite(fileID, IMAGES(:,:,i)', 'float32');
end

fclose(fileID);