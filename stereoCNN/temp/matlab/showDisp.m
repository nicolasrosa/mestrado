clc
clear all
close all

prediction = imread('0000000004.png')
label = imread('0000000004_gt.png')

figure
surf(prediction)
set(gca,'Zdir','reverse')
colormap 'gray'

figure
surf(label)
set(gca,'Zdir','reverse')
colormap 'gray'