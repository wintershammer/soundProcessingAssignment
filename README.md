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

# Experiments

Some example results are:

* Content: `r_S.mp3`, Style `m_S.mp3`, Result: `rs_ms.wav` 
![Results](https://github.com/wintershammer/soundProcessingAssignment/blob/master/figure_1-1.png)
* Content: `r_C.mp3`, Style `m_S.mp3`, Result: `rc_ms.wav` 
![Results](https://github.com/wintershammer/soundProcessingAssignment/blob/master/figure_1-2.png)
* Content: `r_C.mp3`, Style `m_C.mp3`, Result: `rc_mc.wav` 
![Results](https://github.com/wintershammer/soundProcessingAssignment/blob/master/figure_1.png)
* Content: `guitarFolk.wav`, Style `jazzpack.wav`, Result: `jazzFolk.wav` 
![Results](https://github.com/wintershammer/soundProcessingAssignment/blob/master/figure_1-4.png)
* Content: `stranaChopped.mp3`, Style `jazzpack.wav`, Result: `stranaJazzLonger`
![Results](https://github.com/wintershammer/soundProcessingAssignment/blob/master/figure_1-5.png)

# Discussion of the experiments

The experiments were focused on style-transfer between tracks of heavily melodic/harmonic character and tracks of exclusively rythmic/percussive character. Unfortunately, due to lack of resources, the results where quite noisy. Cleaned up (denoised, compressed, equalised) versions are also available under the `experiments` folder.

The experiments where done in two major batches:
* Transfer of melodic style to rythmic content (with results: 'rs_ms.mp3', 'rc_ms.mp3', 'rc_mc.mp3').
* Transfer of rythmic style to melodic content (with results 'jazzFolk.mp3', 'stranaJazzLonger.mp3').

For the first batch we note that the results where essentially the content track (the heavily rythmic one) with a "melody" overlayed on top, following the rythm and tempo of the content. The 'rc_mc' result is particularly interesting, in that it 
contains melodic movement (up and down motion) that was not present in the style track (which exhibited downward motion only).

The second batch of results is much more interesting, showing clear transfer of rythmic elements (especially snare and bass drumwork) onto the content track, with surprising accuracy.

Note that for all experiments, the tempo of the content tracks did not much the tempo of the style ones, yet the results where rythmically accurate.
