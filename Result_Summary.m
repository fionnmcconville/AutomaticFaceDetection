clear all
close all

%Run this script if you want to see a summary of the detector results.

%% Load saved workspace with all end detector results
load('Detector_Result_Summary.mat')

%% Image 1
figure(1),
sgtitle("Image 1 Detection Summary")
subplot(1,3,1)
DrawBoundingBoxes(im1, GGim1detections);
title("Gaussian SVM Gabor");
subplot(1,3,2)
DrawBoundingBoxes(im1, PGim1detections);
title("Polynomial SVM Gabor");
subplot(1,3,3)
DrawBoundingBoxes(im1, GPim1detections);
title("Gaussian SVM PCA");

%% Image 2
figure(2),
sgtitle("Image 2 Detection Summary")
subplot(1,3,1)
DrawBoundingBoxes(im2, GGim2Detections);
title("Gaussian SVM Gabor");
subplot(1,3,2)
DrawBoundingBoxes(im2, PGim2detections);
title("Polynomial SVM Gabor");
subplot(1,3,3)
DrawBoundingBoxes(im2, GPim2detections);
title("Gaussian SVM PCA");

%% Image 3
figure(3),
sgtitle("Image 3 Detection Summary")
subplot(1,3,1)
DrawBoundingBoxes(im3, GGim3detections);
title("Gaussian SVM Gabor");
subplot(1,3,2)
DrawBoundingBoxes(im3, PGim3detections);
title("Polynomial SVM Gabor");
subplot(1,3,3)
DrawBoundingBoxes(im3, GPim3detections);
title("Gaussian SVM PCA");

%% Image 4
figure(4),
sgtitle("Image 4 Detection Summary")
subplot(3,1,1)
DrawBoundingBoxes(im4, GGim4detections);
title("Gaussian SVM Gabor");
subplot(3,1,2)
DrawBoundingBoxes(im4, PGim4detections);
title("Polynomial SVM Gabor");
subplot(3,1,3)
DrawBoundingBoxes(im4, GPim4detections);
title("Gaussian SVM PCA");