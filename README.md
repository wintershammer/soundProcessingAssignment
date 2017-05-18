Neural Musical Style Transfer - Assignment for Sound and Speech Processing course

This is an implementation of musical style transfer, using convolutional neural networks.

# Dependencies
* Numpy
* Tensorflow
* Librosa

# How it works

Given two input sound files, one of which supplies the "content" and the other the "style", the implementation (see `myStyle.py`)
follows the four following steps:

* Load the two input files, perform short-time Fourier transform on them, and return the results after dropping the phases.
* Pass the two transformed files through the convolutional net, and return a "content" feature vector and a "style" Gram matrix.
* Starting with noise as the input to the aforementioned network, optimise based on a loss function that incorporates both the L<sup>2</sup>
distances of the output's feature vector from the "content" feature vector, weighted by a factor of ALPHA, as well as the 
output's gram matrix from the "style" Gram matrix. For the optimiser, I chose L-BFGS, as do most implementations of style-transfer for images.
* Save the resulting optimised input as a ".wav" file, after performing phase reconstruction (see "Signal estimation from modified short-time Fourier transform. D Griffin, J Lim")

A note on Gram matrices: Gram matrices carry information on the linear independence of the vectors they are produced from, and, empirically at least, are a good way to find (pairwise?) "feature correlations".

# Results

Some example results are:

* Content: `r_S.mp3`, Style `m_S.mp3`, Result: `rs_ms.wav` 
![Results](https://github.com/wintershammer/soundProcessingAssignment/blob/master/figure_1-1.png)
* Content: `r_C.mp3`, Style `m_S.mp3`, Result: `rc_ms.wav` 
![Results](https://github.com/wintershammer/soundProcessingAssignment/blob/master/figure_1-2.png)
* Content: `r_C.mp3`, Style `m_C.mp3`, Result: `rc_mc.wav` 
![Results](https://github.com/wintershammer/soundProcessingAssignment/blob/master/figure_1.png)
* Content: `guitarFolk.wav`, Style `jazzpack.wav`, Result: `jazzFolk.wav` 
![Results](https://github.com/wintershammer/soundProcessingAssignment/blob/master/figure_1-4.png)
