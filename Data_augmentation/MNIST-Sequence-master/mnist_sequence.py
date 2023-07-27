from __future__ import print_function
import numpy as np
from load import MNIST
from random import choice


class MNIST_Sequence(object):

    def __init__(self, path='data', name_img='t10k-images.idx3-ubyte',
                 name_lbl='t10k-labels.idx1-ubyte'):
        self.dataset = MNIST(path, name_img, name_lbl)
        self.images, self.labels = self.dataset.load()
        self.label_map = [[] for i in range(10)]
        self.__generate_label_map()
        print('Map Loaded')

    def __calculate_uniform_spacing(self, size_sequence, minimum_spacing, maximum_spacing,
                                    total_width, image_width=28):
        if size_sequence <= 1:
            return 0
        allowed_spacing = (total_width - size_sequence * image_width) / ((size_sequence - 1) * 1.0)
        if not allowed_spacing.is_integer() or allowed_spacing < minimum_spacing \
                or allowed_spacing > maximum_spacing:
            print("Uniform spacing is not possible for the given set of values, " +
                  "please provide suitable values.")
            print("For example, try with sequence [0, 1] with minimum spacing 0, " +
                  "maximum_spacing 10 and image_width 66.")
            exit()
        return int(allowed_spacing)

    def __generate_label_map(self):
        num_labels = len(self.labels)
        for i in range(num_labels):
            self.label_map[self.labels[i]].append(i)

    def __select_random_label(self, label):
        if len(self.label_map[label]) > 0:
            return choice(self.label_map[label])
        else:
            print("No images for the number " + str(label) +
                  " is available. Please try with a different number.")
            exit()

    # def generate_image_sequence(self, sequence, minimum_spacing, maximum_spacing,
    #                             total_width, image_height=28):
    def generate_image_sequence(self, sequence, allowed_spacing, image_height=28):
        sequence_length = len(sequence)
        # allowed_spacing = self.__calculate_uniform_spacing(sequence_length, minimum_spacing,
        #                                                    maximum_spacing, total_width)
        # allowed_spacing = 5
        # spacing = np.ones(image_height * allowed_spacing,
        #                   dtype='float32').reshape(image_height, allowed_spacing)
        spacing = np.ones(image_height * allowed_spacing, dtype='float32').reshape(allowed_spacing, image_height)
        whole_image = ''
        for i in range(sequence_length):
            random_label_number = self.__select_random_label(sequence[i])
            if i < 1:
                dataset_image = self.images[random_label_number]
                whole_image = np.vstack((dataset_image, spacing))
            elif i < sequence_length-1:
                dataset_image = self.images[random_label_number]
                temp_image = np.vstack((dataset_image, spacing))
                whole_image = np.vstack((whole_image, temp_image))
            else:
                dataset_image = self.images[random_label_number]
                whole_image = np.vstack((whole_image, dataset_image))
            # random_label_number = self.__select_random_label(sequence[i])
            # image = np.vstack((image, self.images[random_label_number]))
        return whole_image
