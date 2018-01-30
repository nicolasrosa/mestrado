# ===========
#  Libraries
# ===========
import os

from utils.monodeep_model import *


# ===================
#  Class Declaration
# ===================
class ImportGraph(object):
    """  Importing and running isolated TF graph """

    def __init__(self, restore_path):
        # Get the Path of the Model to be Restored
        restore_files = os.listdir(restore_path)
        assert len(
            restore_files) == 4, "Houston we've got a problem. 'restore_path' specified should have only the files 'checkpoint', '*.ckpt.data-00000-of-00001', '*.ckpt.index' and '*.ckpt.meta'."
        # print(restore_files)
        for file in restore_files:
            # print(file)
            # print(file.find("model"))
            if not file.find("model"):
                model_name = file.split(".")[1]  # '*.ckpt.meta' file
                model_fileName = os.path.splitext(file)[0]
                restore_filepath = restore_path + model_fileName  # Path to file with extension *.ckpt
                break

        # print(args.model_name)
        # print(model_fileName)
        # print(restore_filepath)

        # Creates local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'restore_filepath' into local graph
            print('\n[Network/Restore] Restoring model from file: %s' % restore_filepath)
            saver = tf.train.import_meta_graph(restore_filepath + '.meta', clear_devices=True)
            saver.restore(self.sess, restore_filepath)
            print("[Network/Restore] Model restored!")
            print("[Network/Restore] Restored variables:\n", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), '\n')

            # Gets activation function from saved collection
            # You may need to change this in case you name it differently
            self.tf_image = tf.get_collection('image')[0]
            self.tf_predCoarse = tf.get_collection('predCoarse')[0]
            self.tf_predFine = tf.get_collection('predFine')[0]
            self.tf_keep_prob = tf.get_collection('keep_prob')[0]

    def networkPredict(self, image):
        """ Running the activation function previously imported """
        # The 'inputs' corresponds to name of input placeholder
        data = np.expand_dims(image, 0)  # (idx, height, width, numChannels) - Normalized
        feed_dict = {self.tf_image: data, self.tf_keep_prob: 1.0}

        # ----- Session Run! ----- #
        predCoarse, predFine = self.sess.run([self.tf_predCoarse, self.tf_predFine], feed_dict=feed_dict)
        # -----

        return predCoarse, predFine
