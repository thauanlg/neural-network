"""
 In this program has been implemented a cnn arquitecture to train and test the CIFAR-10 dataset. It's been used the mnist tutorial
 from tensorflow which gave us an idea of how to implement a acceptable cnn arquitecture. The pdf description provides the references
 and links to get the dataset and the tutorial.
"""
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image

tf.logging.set_verbosity(tf.logging.ERROR)

# Defining the models used
def cnn_model_fn(features, labels, mode):
    inRed, inGreen, inBlue = tf.split(features["x"], num_or_size_splits=3, axis=1)
    inRed = tf.reshape(inRed, shape=[-1, 32, 32, 1])
    inGreen = tf.reshape(inGreen, shape=[-1, 32, 32, 1])
    inBlue = tf.reshape(inBlue, shape=[-1, 32, 32, 1])

    # Convolutional Layer #1
    conv1Red = tf.layers.conv2d(
        inputs=inRed,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
  
    conv1Green = tf.layers.conv2d(
        inputs=inGreen,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
  
    conv1Blue = tf.layers.conv2d(
        inputs=inRed,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
  
    # Pooling Layer #1
    pool1Red = tf.layers.max_pooling2d(inputs=conv1Red, pool_size=[2, 2], strides=2)
    pool1Green = tf.layers.max_pooling2d(inputs=conv1Green, pool_size=[2, 2], strides=2)
    pool1Blue = tf.layers.max_pooling2d(inputs=conv1Blue, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2Red = tf.layers.conv2d(
        inputs=pool1Red,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    conv2Green = tf.layers.conv2d(
        inputs=pool1Green,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    conv2Blue = tf.layers.conv2d(
        inputs=pool1Blue,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    pool2Red = tf.layers.max_pooling2d(inputs=conv2Red, pool_size=[2, 2], strides=2)
    pool2Green = tf.layers.max_pooling2d(inputs=conv2Green, pool_size=[2, 2], strides=2)
    pool2Blue = tf.layers.max_pooling2d(inputs=conv2Blue, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    pool2_flatRed = tf.reshape(pool2Red, [-1, 8 * 8 * 64])
    pool2_flatGreen = tf.reshape(pool2Green, [-1, 8 * 8 * 64])
    pool2_flatBlue = tf.reshape(pool2Blue, [-1, 8 * 8 * 64])

    pool2_flat = tf.concat([pool2_flatRed, pool2_flatGreen, pool2_flatBlue], axis=1)

    # Dense Layer
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.7 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Function that train the model
def train(mnist_classifier):
    dict=pickle.load(open("./cifar/data_batch_1", "rb"), encoding="bytes")
    train_images=np.asarray(dict[b'data'], dtype=np.float32)
    train_labels = np.array(dict[b"labels"], dtype=np.int32)

    for i in range(2,6):
        dict=pickle.load(open("./cifar/data_batch_"+str(i), "rb"), encoding="bytes")
        train_images_aux=np.asarray(dict[b'data'], dtype=np.float32)
        train_labels_aux=np.array(dict[b"labels"], dtype=np.int32)
        train_images=np.r_[train_images, train_images_aux]
        train_labels = np.r_[train_labels, train_labels_aux]
  
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_images},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=100000)

# Function that evaluates the model, using the test batch from the CIFAR-10 dataset
def eval(mnist_classifier):
    dict=pickle.load(open("cifar/test_batch", "rb"), encoding="bytes")
    test_images=np.asarray(dict[b'data'], dtype=np.float32)
    test_labels = np.array(dict[b"labels"], dtype=np.int32)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_images},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print("Acurácia: {:.2f}".format(eval_results['accuracy']))

# Function that evaluates the model, using jpeg image for it
def testImage(mnist_classifier):
    labelNames = ['airplane', 'automobile', 'bird',  'cat', 'deer', 
                  'dog',  'frog',  'horse',  'ship',   'truck']
    testImage = [0]*10
    for i in range(10):
        img = Image.open("./images/"+labelNames[i]+".jpeg")
        img = np.asarray(img.resize((32, 32), Image.ANTIALIAS))
        testImage[i]=img.reshape((32*32,3)).T.reshape(32*32*3)
    testImage=np.asarray(testImage, dtype=np.float32)
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": testImage},
        y=np.arange(10),
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print("Acurácia: {:.2f}".format(eval_results['accuracy']))

# This function asks for training, testing and testing with images provides in image directory
def main(args):
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/home/thauan/Documents/8º Semestre/Redes Neurais/projeto2/save")

    choice = int(input("1 : Treinar\n2 : Avaliar o lote de teste\n3 : Avaliar as imagens de teste\n"))
    if choice == 1:
        train(mnist_classifier)
    elif choice == 2:
        eval(mnist_classifier)
    else:
        testImage(mnist_classifier)

if __name__ == "__main__":
    tf.app.run()
