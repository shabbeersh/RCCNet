%The following code is written by Shiv Ram Dubey, Computer Vision Group, IIIT Sri City, India
%This code extracts the patches of Colon Cancer Dataset available from https://warwick.ac.uk/fac/sci/dcs/research/tia/data/crchistolabelednucleihe/
%In case you are using this code, please cite the following paper:
%Basha, SH Shabbeer, et al. "Rccnet: An efficient convolutional neural network for histological routine colon cancer nuclei classification." 2018 15th International Conference on Control, Automation, Robotics and Vision (ICARCV). IEEE, 2018.

function ColonCancerPatchExtraction()
in_path='CRCHistoPhenotypes_2016_04_28/Classification/';
out_path='crchistophenotypes32_32/';
if ~exist([out_path 'epithelial'])
    mkdir([out_path 'epithelial'])
end
if ~exist([out_path 'fibroblast'])
    mkdir([out_path 'fibroblast'])
end
if ~exist([out_path 'inflammatory'])
    mkdir([out_path 'inflammatory'])
end
if ~exist([out_path 'others'])
    mkdir([out_path 'others'])
end
for i=1:100
    folder=['img' num2str(i)];
    im=imread([in_path 'img' num2str(i) '/img' num2str(i) '.bmp']);
    
    a=load([in_path 'img' num2str(i) '/img' num2str(i) '_epithelial']);a=a.detection;
    for j=1:size(a,1)
       im1=imcrop(im,[a(j,1)-16,a(j,2)-16,31,31]);
       imwrite(imresize(im1,[32 32]),[out_path 'epithelial/img' num2str(i) '_' num2str(j) '.jpeg']);
    end
    a=load([in_path 'img' num2str(i) '/img' num2str(i) '_fibroblast']);a=a.detection;
    for j=1:size(a,1)
       im1=imcrop(im,[a(j,1)-16,a(j,2)-16,31,31]);
       imwrite(imresize(im1,[32 32]),[out_path 'fibroblast/img' num2str(i) '_' num2str(j) '.jpeg']);
    end
    a=load([in_path 'img' num2str(i) '/img' num2str(i) '_inflammatory']);a=a.detection;
    for j=1:size(a,1)
       im1=imcrop(im,[a(j,1)-16,a(j,2)-16,31,31]);
       imwrite(imresize(im1,[32 32]),[out_path 'inflammatory/img' num2str(i) '_' num2str(j) '.jpeg']);
    end
    a=load([in_path 'img' num2str(i) '/img' num2str(i) '_others']);a=a.detection;
    for j=1:size(a,1)
       im1=imcrop(im,[a(j,1)-16,a(j,2)-16,31,31]);
       imwrite(imresize(im1,[32 32]),[out_path 'others/img' num2str(i) '_' num2str(j) '.jpeg']);
    end
end
end
