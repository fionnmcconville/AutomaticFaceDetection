close all
clear all

%% Loading in images
im1 = imread('im1.jpg');
im2 = imread('im2.jpg');
im3 = imread('im3.jpg');
im4 = imread('im4.jpg'); %Takes a long while, even on polynomial
im5 = imread('im5.jpg'); %New test image


%% Detections - Testing Images

%% Image 1
%Initial Detections
tic
figure,
subplot(2,2,1), imagesc(im1), title("Sliding Window Processing"),
colormap(gray), drawnow, hold on;
im1Detections = SlidingWindow(im1);
hold off;
DrawBoundingBoxes(im1, im1Detections);


% Non maxima Supression

subplot(2,2,2),
im1Detections = simpleNMS(im1Detections);
DrawBoundingBoxes(im1, im1Detections);
title("Im1 Non-Maxima Supression");
toc

%% Image 2
tic

%Initial Detections
subplot(2,2,3), imagesc(im2), title("Sliding Window Processing"),
colormap(gray), drawnow, hold on;
im2Detections = SlidingWindow(im2);
hold off;
DrawBoundingBoxes(im2, im2Detections);

%Non maxima Supression
im2Detections = simpleNMS(im2Detections);
subplot(2,2,4),
DrawBoundingBoxes(im2, im2Detections);
title("Im2 Non-Maxima Supression");
toc

%% Image 3
tic
%Initial Detections
figure(2),
subplot(2,2,1), imagesc(im3), title("Sliding Window Processing"),
colormap(gray), drawnow, hold on;
im3Detections = SlidingWindow(im3);
hold off;
DrawBoundingBoxes(im3, im3Detections);

% Non maxima Supression
im3Detections = simpleNMS(im3Detections, 0.2);
subplot(2,2,2),
DrawBoundingBoxes(im3, im3Detections);
title("Im3 Non-Maxima Supression");

toc

%% Image 4
tic

%Initial Detections
figure(2),
subplot(2,2,3), imagesc(im4), title("Sliding Window Processing"),
colormap(gray), drawnow, hold on;
im4Detections = SlidingWindow(im4);
hold off;
DrawBoundingBoxes(im4, im4Detections);

% Non maxima Supression
im4Detections = simpleNMS(im4Detections, 0.2);
subplot(2,2,4),
DrawBoundingBoxes(im4, im4Detections);
title("Im4 Non-Maxima Supression")
toc


%% Detections - Test new image
%Here we include a new image to the testing set to see if the model
%performs well on any image
tic
figure(3),
imshow(im5), colormap(gray), drawnow, hold on;
%Initial Detections
detections = SlidingWindow(im5);
hold off;
subplot(1,2,1),
DrawBoundingBoxes(im5, detections);
title("Detections")
hold off;

% Non maxima Supression
detections = simpleNMS(detections, 0.2);
subplot(1,2,2),
DrawBoundingBoxes(im5, detections);
title("Non-Maxima Supression")
toc
