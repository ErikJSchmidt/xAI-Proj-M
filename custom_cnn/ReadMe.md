# Custom CNNs
As first step of the project we implement CNN models with pytorch and explore effects of different model shapes.

## Plain18Layer
We follow descriptions of "Deep Residual Learning for Image Recognition" as closely to construct the plain 18 layer
network they described.

Descriptions from the paper:
1. _The convolutional layers mostly have 3×3 filters and
   follow two simple design rules: (i) for the same output
   feature map size, the layers have the same number of filters; and (ii) if the feature map size is halved, the number 
of filters is doubled so as to preserve the time complexity per layer._
2. _We perform downsampling directly by
   convolutional layers that have a stride of 2_
3. _The network
   ends with a global average pooling layer and a 1000-way
   fully-connected layer with softmax._
4. _We adopt batch
   normalization (BN) [16] right after each convolution and
   before activation, following [16]._
5. _We initialize the weights
   as in [13] and train all plain/residual nets from scratch._
6. _We use SGD with a mini-batch size of 256._
7. _The learning rate
   starts from 0.1 and is divided by 10 when the error plateaus,
   and the models are trained for up to 60 × 104
   iterations._
8. _We use a weight decay of 0.0001 and a momentum of 0.9._
9. _We do not use dropout [14], following the practice in [16]._

#### ToDo:
- downsampling with stride 2

#### Effect
- After changing last layer, still no learning
- custom_cnn/savedmodels/Plain18Layer20231130_19:29: After add batchnorm, the model improves on the baseline acc and learns throught the 20 epochs we tried