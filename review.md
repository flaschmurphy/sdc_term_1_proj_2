## Design and Test a Model Architecture

The submission describes the preprocessing techniques used and why these techniques were chosen.
Well done implementing data preprocessing, here is a Pro Tip for you regarding preprocessing and augmentation, though you have already implemented some of the suggested options.

Advanced Pro tip: Data preprocessing and augmentation.

After inspecting our data, before engaging with the network architecture it is advisable to proactively perform some data augmentation (in deep learning more data is always good) and some data preprocessing.

## Data augmentation:
Our dataset might not be as big as we wish or we might want to improve the generalisation ability of our network by augmenting our dataset this can be achieved through several approaches aimed at exploiting the data we already have by augmenting it, here are some ways to do that:

Rotating our data: We could perform random rotations of our images, possibly drawing the rotation angle from a normal distribution to avoid biasing the newly created images.
Flipping and cropping images (preserving the relevant parts of the image)
Adding noise or filter.
When augmenting data it will be preferable to improve the number of images in those cases that are underrepresented in the dataset.
Here is a cool Python library to deal with those issues: https://github.com/aleju/imgaug

## Data preprocessing

We can use several techniques to preprocess our data:

- Normalization
- Standardization
- Brightness augmentation
- Histogram Equalization (Python resources: http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)
- PCA/ZCA whitening (This is well beyond the course and requirements, please consider it only for your information.)

More resources on data preprocessing and augmentation:

  http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html (this link
  contains very cool tricks and tips that extend beyond preprocessing and
  augmentation)
- On data preprocessing: http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing
- Sci-Kit image processing: http://scikit-image.org/docs/dev/api/skimage.transform.html
- The submission provides details of the characteristics and qualities of the
  architecture, including the type of model used, the number of layers, and the
  size of each layer. Visualizations emphasizing particular qualities of the
  architecture are encouraged.

Architecture

Well done discussing the characteristics of your network, visualising it would
surely make it easier for the reader to really grasp the architecture and the
rationale behind it. There are several options to achieve that:

1. When using Tensor Flow you could take advantage of TensorBoard, an extremely
   powerful tool available to Tensor Flow users:
   https://www.tensorflow.org/get_started/graph_viz Here are are a couple of
   places to get started: https://github.com/PythonWorkshop/tensorboard_demos
   https://ischlag.github.io/2016/06/04/how-to-use-tensorboard/ 
2. Use a third-party tool like: https://github.com/bruckner/deepViz 
3. You can check this article for known automated/integrated drawing tools:
   https://www.reddit.com/r/MachineLearning/comments/574usi/discussion_what_do_you_use_for_neural_network/

## The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

Excellent job implementing dropout, that is an effective way to avoid
overfitting!  #### Advanced Pro tip: Model training Model training is quite an
art, there is no static rule though it is helpful to get some guidance as there
are quite a few elements to consider. The following source is incredibly
helpful in thoroughly framing the problem in providing some initial guidance:
http://machinelearningmastery.com/improve-deep-learning-performance/ Among
others the following elements regarding training are addressed in the “Improve
Performance With Algorithm Tuning” section:

- Diagnostics.
- Weight Initialization.
- Learning Rate.
- Activation Functions.
- Network Topology.
- Batches and Epochs.
- Regularization.
- Optimization and Loss.
- Early Stopping.
- You might be interested in the one in bold. As mentioned above this is just a
  section of this invaluable post where you can find advice on many other issues
  related to improving your network.
- You might as well be interested in the part discussing the model’s
  architecture, please pay particular attention to the first advice! :)

How many layers and how many neurons do you need? No one knows. No one. Don’t ask.
- Try one hidden layer with a lot of neurons (wide).
- Try a deep network with few neurons per layer (deep).
- Try combinations of the above.
- Try architectures from recent papers on problems similar to yours. I’m
  reposting a couple of links you might find useful:
  https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad#.fqpfkvjak
- https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc#.3z22vq3xl
- Try topology patterns (fan out then in) and rules of thumb from books and papers.


