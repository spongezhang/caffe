dir = '../../data/ukbench/images/';
for i = 0:4:10196
    if mod(i,1000)==0
      disp(i);
    end
    filename = sprintf('ukbench%05d.jpg',i);
    savefilename = sprintf('ukbench%05d_0.jpg',i);
    im = imread([dir filename]);
    index = rand(1,5);
    angle = index(1)*360;
    scale = 1+0.4*(index(2)-0.5);
    tim = imrotate(im,angle,'bilinear','crop');
    %imshow(tim);
    imwrite(tim,[dir savefilename]);
    %pause;
    %disp(size(im)-size(tim));
    
end
