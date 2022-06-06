clf

f=dir('img/*.jpg');
files={f.name};
imgs=cell(numel(files), 1);

% structure element for eroding
ser=strel('disk', 5);

for k=1:numel(files)
    imgs{k}=imread(append('img/', files{k}));

    tmp=rgb2gray(imgs{k});

    % maybe set treshold manually
    bw=imbinarize(tmp);
    bw=imerode(bw, ser);
    [L, n] = bwlabel(bw);
    
    subplot(2, 6, 1); imshow(L);
    title(['dice' num2str(k) '.jpg N=' num2str(n)]);

    % crop image

    for j=1:n
        [r, c] = find(L==j);

        sr = min(r);
        br = max(r);
        sc = min(c);
        bc = max(c);

        die=L(sr:br, sc:bc);
        die=~die;
        die=imclearborder(die);

        [centers, radii] = imfindcircles(die, [15,30], 'Sensitivity',0.90);

        subplot(2, 6, j+6); imshow(die);
        h=viscircles(centers, radii);
    end
    figure;

    % diameter is about 30 to 50 +-
    % 39.4
    
end
