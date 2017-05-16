import tensorflow as tf
import librosa
import numpy as np

    
def generateFourierMagns(filename, fourierWindowSize):
    x, fs = librosa.load(filename, duration=10.0) #load 5 secs
    #remember: stft returns D[f,t] = frequency @ time t
    #abs(D[f,t]) = magnitude of frequency f at time t
    #angle(D[f,t]) = phase of frequency f at time t
    stftMagns = librosa.stft(x, fourierWindowSize)
    print("input sampled @",fs)

def runNetwork(channelNum, sampleNum, kernel, optimisationVars):
    
    #construct convlutionalNet
    graph = tf.Graph()
    with graph.as_default():
        #define an initial noisy var, (to be replaced by input content/style vectors when generating style/content)
        #myInVar = tf.Variable(np.random.randn(1,1,sampleNum,channelNum).astype(np.float32)*1e-3, name="myInVar")

        myInVar = tf.placeholder('float32', [1,1,sampleNum,channelNum], name="myInVar")
        
        #define the kernel as a constant
        kernelConst = tf.constant(kernel, name="kernel", dtype="float32")

        #define the convolution layer
        conv = tf.nn.conv2d(myInVar, kernelConst, strides = [1,1,1,1], padding="VALID",name="convLayer")

        #define rectified linear units layer which takes the results of conv
        net = tf.nn.relu(conv,name="finalLayer")
        
        #note that I didn't use any fully-connected layers, following the paradigm set by the original neural-style authors
        #the fully-conned layers are usually used for classification and so I have no use for them (i'm not classifying anything)
        #might be cool to add a fully-conned layer tho, see : http://liipetti.net/erratic/2016/03/28/controlling-image-content-with-fc-layers/

    return graph

def generateStyleAndContent(content, style, kernel):
    
    channelNumContent = content.shape[0] #number of "channels" (i.e freq bins)
    sampleNumContent = content.shape[1] #number of "samples" (i.e values @ time steps)

    channelNumStyle = style.shape[0]
    sampleNumStyle = style.shape[1]

    
    #remember, tf convNets expects images, with shape [batch, in_height, in_width, in_channels]
    contentFormatted = np.ascontiguousarray(content.T[None,None,:,:]) #convert them to tensorFlow specifications
    styleFormatted = np.ascontiguousarray(style.T[None,None,:,:])

    with tf.Session(graph=runNetwork(channelNumContent,sampleNumContent,kernel,[])) as sess:
        netaki = sess.graph.get_operation_by_name('finalLayer').outputs[0]
        contentFeatures = sess.run(netaki, feed_dict={"myInVar:0": contentFormatted})
        styleFeatures = sess.run(netaki, feed_dict={"myInVar:0": styleFormatted})
        reshaped = np.reshape(styleFeatures, (-1, filterNum)) #reshape, drops the batch/in_height dims
        styleGramMat = np.matmul(reshaped.T , reshaped) / sampleNumContent #gram matrix for style

    return contentFeatures, styleGramMat

def styleTransfer(contentFeats, styleGram, kernel, alpha, iterations, channelNum, sampleNum, filterNum):
    with tf.Graph().as_default():
        myInVar = tf.Variable(np.random.randn(1,1,sampleNum,channelNum).astype(np.float32)*1e-3, name="myInVar") #start with noise

        kernelConst = tf.constant(kernel, name="kernel", dtype='float32')

        factor = np.sqrt(2) * np.sqrt(2.0 /  ((channelNum + filterNum) * 11))
        kernel = np.random.randn(1, 11, channelNum, filterNum) * factor
        
        conv = tf.nn.conv2d(myInVar, kernelConst, strides=[1, 1, 1, 1], padding="VALID", name="convLayer")

        net = tf.nn.relu(conv)

        reshaped = tf.reshape(net, (-1, filterNum))
        outGram = tf.matmul(tf.transpose(reshaped), reshaped) / sampleNum


        content_loss = alpha * 2 * tf.nn.l2_loss(net - contentFeats) #alpha times L2 distance as loss betwene feature vecs
        style_loss = 2  * tf.nn.l2_loss(outGram - styleGram) #L2 distance as loss between the two gram matrices
        
        loss = content_loss + style_loss #loss is combination of both

        print("Starting optimisation run")
        optimiser = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': iterations})

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            optimiser.minimize(sess)
    
            print ('Final loss:', loss.eval())
            generatedMusic = myInVar.eval()

       

    return generatedMusic

fs = 0   

filterNum = 4096
#defining filter (gotta be the same for all runs!)
factor = np.sqrt(2) * np.sqrt(2.0 /  ((content.shape[0] + filterNum) * 11))
kernel = np.random.randn(1, 11, content.shape[0], filterNum) * factor


contentFeatures, styleGramMat = generateStyleAndContent(content, style, kernel)

a = np.zeros_like(content)

# phase reconstruction, see griffin-lim
p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(500):
    S = a * np.exp(1j*p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, 2048))

librosa.output.write_wav("rc_mc.wav", x, fs)

