import numpy as np
import cv2

import tensorflow as tf

from tensorflow.core.framework.graph_pb2 import GraphDef

# Network I/O
INPUT_NAMES = ["Placeholder:0", "Placeholder_1:0", "Placeholder_2:0", "Placeholder_3:0", "Placerholder_4:0"]
OUTPUT_NAMES = ["FusionLayer_B_0_2/add_9:0", "Select:0"]

class ObstructionNetwork():
    """
    Class returns output images from obstruction removal network.
    """

    def __init__(self, pb_path):
        """
        Creates tf function for neural network.
        """
        with open(pb_path, "rb") as pb:
            graph_def = GraphDef()
            graph_def.ParseFromString(pb.read())

        @tf.function
        def network_function(I0, I1, I2, I3, I4):
            # inputs = {}
            # input_images = [I0, I1, I2, I3, I4]
            # for idx in range(5):
            #     inputs[INPUT_NAMES[idx]] = input_images[idx]

            inputs = {"Placeholder:0":I0, "Placeholder_1:0":I1, "Placeholder_2:0":I2, "Placeholder_3:0":I3, "Placeholder_4:0":I4}
            outputs = ["FusionLayer_B_0_2/add_9:0", "Select:0"]
            alpha, background = tf.graph_util.import_graph_def(graph_def,
                    input_map=inputs,
                    return_elements=outputs)
            return alpha, background

        self._network = network_function 

    def run(self, images):
        """
        Runs network with an error check for only five images in the inputs.
        """
        if len(images) != 5:
            raise Exception("Network needs exactly five images")

        input_proc, output_proc = generate_image_processors(images[0])
        proc_images = list(map(input_proc, images))
        alpha, background = self._network(*proc_images)
        return output_proc(alpha), output_proc(background)

# Helper functions
def generate_image_processors(I0):
    """
    Generates image pre and postprocessing functions.
    """
    normal_i0 = I0.astype(np.float32)[..., ::-1] / 255.0

    ORIGINAL_H = normal_i0.shape[0]
    ORIGINAL_W = normal_i0.shape[1]
    orig_size = (ORIGINAL_W, ORIGINAL_H)

    RESIZED_H = int(np.ceil(float(ORIGINAL_H) * 1.0 / 16.0))*16
    RESIZED_W = int(np.ceil(float(ORIGINAL_W) * 1.0 / 16.0))*16
    new_size = (RESIZED_W, RESIZED_H)

    def input_proc(img):
        """
        Mimics input processing from original repo.
        """
        norm = img.astype(np.float32)[..., ::-1] / 255.0
        resized = np.expand_dims(
                cv2.resize(norm, dsize=new_size, interpolation=cv2.INTER_CUBIC), 
                0)
        return resized

    def output_proc(img):
        """
        Mimics outpu processing from original repo.
        """
        resized = cv2.resize(img.numpy()[0, :, :, ::-1], orig_size, interpolation=cv2.INTER_CUBIC)
        rounded = np.round(resized * 255.0)
        clipped = np.clip(rounded, 0.0, 255.0).astype(np.uint8) 
        return clipped

    return input_proc, output_proc
