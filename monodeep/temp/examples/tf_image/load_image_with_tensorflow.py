# Typical setup to include TensorFlow.
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Make a queue of file names including all the JPEG/PNG images files in the relative image directory.
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("images/jpg/*.jpg"))
# filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("images/png/*.png"))

# Read an entire image file which is required since they're JPEGs, if the images are too large they could be split in
# advance to smaller files or use the Fixed reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG/PNG file, this will turn it into a Tensor which we can then use in training.
tf_image = tf.image.decode_jpeg(image_file)
# tf_image = tf.image.decode_png(image_file) # use png or jpg decoder based on your files.

# Resizes the image to size 224x224 (height, width)
tf_resized_image=tf.image.resize_images(tf_image, [224, 224]) # Default: Bilinear interpolation.

# TODO: Implemente batch
# tf_image.set_shape((224, 224, 3))
# batch_size = 50
# num_preprocess_threads = 1
# min_queue_examples = 256
# images = tf.train.shuffle_batch(
# [tf_image],
# batch_size=batch_size,
# num_threads=num_preprocess_threads,
# capacity=min_queue_examples + 3 * batch_size,
# min_after_dequeue=min_queue_examples)

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    # TODO: Como pegar o tamanho da lista?
    for i in range(3):  # length of your filename list
        # image = sess.run([tf_image])
        image = tf_image.eval() # Run your image Tensor :)
        # resized = tf_resized_image.eval()

        # print("image:", image)
        # print(image.shape)
        # print(image.dtype)

        # print("resized:", resized)
        # print(resized.shape)
        # print(resized.dtype)

        plt.figure(1)
        plt.imshow(image)
        # plt.figure(2)
        # plt.imshow(resized[:,:,1]) # FIXME: Não é possível mostrar todos os canais. Bug: plt.imshow(resized)
        plt.pause(0.5)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

print("Done.")