# Glottis-segmentation-DNN
This is the repository to support the publication "Automatic glottis localization and segmentation in stroboscopic videos using deep neural network"

Code expects the training data to be prepared before (input: Nx27 and output Nx1)
The input is the 3x3 neighborhood data for the color rgb pixel result in a 27 dim vector and the label of the center pixel (if its glottis pixel-->1 or 0--> otherwise)
