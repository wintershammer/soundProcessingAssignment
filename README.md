Neural Musical Style Transfer - Assignment for Sound and Speech Processing course

This is an implementation of musical style transfer, using convolutional neural networks.

#Dependencies
* Numpy
* Tensorflow
* Librosa

#How it works

Given two input sound files, one of which supplies the "content" and the other the "style", the implementation (see `mystyle.py`)
follows the four following steps:

* Load the two input files, perform short-time Fourier transform on them, and return the results after dropping the phases.
* Pass the two transformed files through the convolutional net, and return a "content" feature vector and a "style" gram matrix.
* Starting with noise as the input to the aforementioned network, optimise based on a loss function that incorporates both the L^2 
distances of the output's feature vector from the "content" feature vector, weighted by a factor of ALPHA, as well as the 
output's gram matrix from the "style" gram matrix.
* Save the resulting optimised input as a ".wav" file, after performing phase reconstruction (see "Signal estimation from modified short-time Fourier transform. D Griffin, J Lim")


