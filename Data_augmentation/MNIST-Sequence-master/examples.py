from __future__ import print_function
from mnist_sequence_api import MNIST_Sequence_API


api_object = MNIST_Sequence_API()

# digits (0~9), ex: range(8)
# spacing_range, ex: (0, 10)
# image_width (pixel): 224

sequence = [3, 6, 9, 7, 5]
# api_object.save_image(api_object.generate_mnist_sequence(range(9), 10, 28), range(9))
api_object.save_image(api_object.generate_mnist_sequence(sequence, 10, 28), range(9))

# api_object.save_image(api_object.generate_mnist_sequence(range(8), (0, 10), 224), range(8))
# api_object.save_image(api_object.generate_mnist_sequence(range(9), (0, 10), 292), range(9))
# api_object.save_image(api_object.generate_mnist_sequence(range(10), (0, 10), 280), range(10))
