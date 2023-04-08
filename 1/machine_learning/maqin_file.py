import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from PIL import Image

import time
import io

MAX_IMAGE_DIMENSIONS = 256
TOTAL_VARIATION_WEIGHT = 500
PROCESSING_STEPS = 400

content_img = Image.open("/Users/err/PycharmProjects/АИСД/machine_learning/Mashine_learning/IMG_3540.JPG")

'''
def img_scaler(image, max_dim=256):
    # Превращает тензор в новый тип.
    original_shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    # Создает константу масштаба для изображения
    scale_ratio = 4 * max_dim / max(original_shape)
    # Превращает тензор в новый тип.
    new_shape = tf.cast(original_shape * scale_ratio, tf.int32)
    # Изменяет размер изображения на основе константы масштабирования, созданной выше
    return tf.image.resize(image, new_shape)
'''


def rescale_image(image, max_dim):
    w = image.size[0]
    h = image.size[1]
    if w > h:
        w = max_dim
        h = (image.size[1] / image.size[0]) * max_dim
    else:
        h = max_dim
        w = (image.size[0] / image.size[1]) * max_dim

    return image.resize((int(w), int(h)), Image.Resampling.LANCZOS)


def tensorify_image(image):
    bytes_io = io.BytesIO()
    image.save(bytes_io, format="JPG")
    byte_array = bytes_io.getvalue()

    image = tf.image.decode_image(tf.constant(byte_array), channels=3)
    # Convert image to dtype, scaling (MinMax Normalization) its values if needed.
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Adds a fourth dimension to the Tensor because
    # the model requires a 4-dimensional Tensor
    return image[tf.newaxis, :]


class ImageStyleExtractor:
    def __init__(self):
        self.content_layers = ["block5_conv2"]
        self.style_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1"
        ]

        vgg = keras.applications.vgg19.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in (self.style_layers + self.content_layers)]
        self.model = keras.Model([vgg.input], outputs)

        self.opt = tf.optimizers.Adam(learning_rate=0.005, beta_1=0.99, epsilon=1e-1)
        self.style_weight = 1e-2
        self.content_weight = 1e4

        self.style_targets = None
        self.content_targets = None

    def gram_matrix(input_tensor):
        # Tensor contraction over specified indices and outer product.
        # Matrix multiplication
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        # Save the shape of the input tensor
        input_shape = tf.shape(input_tensor)
        # Casts a tensor to a new type.
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        # Divide matrix multiplication output to num_locations
        return result / (num_locations)

    def extract(self, inputs):
        # Process the image input
        inputs = inputs * 255.0
        preprocessed_input = keras.applications.vgg19.preprocess_input(inputs)

        # Feed the preprocessed image to the VGG19 model
        outputs = self.model(preprocessed_input)
        # Separate style and content outputs
        style_outputs, content_outputs = (outputs[:len(self.style_layers)], outputs[len(self.style_layers):])
        # Process style output before dict creation
        style_outputs = [ImageStyleExtractor.gram_matrix(style_output) for style_output in style_outputs]

        # Create two dicts for content and style outputs
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {"content": content_dict, "style": style_dict}

    def style_content_loss(self, outputs):
        style_outputs = outputs["style"]
        content_outputs = outputs["content"]
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / len(self.style_layers)

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / len(self.content_layers)
        loss = style_loss + content_loss
        return loss


per_style_extractors = dict()


def load_all_styles():
    if os.path.isdir("Mashine_learning/styles/"):
        for style in os.listdir("Mashine_learning/styles/"):
            if os.path.isdir("Mashine_learning/styles/" + style + "/"):
                load_style(style)


def load_style(style):
    new_extractors = []

    print("Loading style: " + style, flush=True)

    for style_file in os.listdir("Mashine_learning/styles/" + style + "/"):
        if style_file.endswith(".jpg".lower()):
            image = Image.open("Mashine_learning/styles/" + style + "/" + style_file)
            image = rescale_image(image, MAX_IMAGE_DIMENSIONS)
            image = tensorify_image(image)

            new_extractor = ImageStyleExtractor()
            new_extractor.style_targets = new_extractor.extract(image)["style"]
            new_extractors.append(new_extractor)

    per_style_extractors[style] = new_extractors

    print("Loaded style: " + style, flush=True)


class AIStyleMissingException(Exception):
    pass


def process_image(input_image, style):
    if style not in per_style_extractors:
        raise AIStyleMissingException(style)

    input_image = rescale_image(input_image, MAX_IMAGE_DIMENSIONS)
    input_image = tensorify_image(input_image)

    extractors = per_style_extractors[style]
    style_sample_count = len(extractors)

    steps = 0
    start_time = time.time()
    end_time = time.time()

    for extractor in extractors:
        extractor.content_targets = extractor.extract(input_image)["content"]

        def train_step(image):
            with tf.GradientTape() as tape:
                outputs = extractor.extract(image)
                loss = extractor.style_content_loss(outputs)
                loss += TOTAL_VARIATION_WEIGHT * tf.image.total_variation(image)

            grad = tape.gradient(loss, image)
            extractor.opt.apply_gradients([(grad, image)])
            image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

        image = tf.Variable(input_image)
        for i in range(PROCESSING_STEPS):
            train_step(image)

            end_time = time.time()

            _str = "Processing" + "." * (steps % 4) + " " * (6 - (steps % 4))
            _str += "[" + str(int(steps / (PROCESSING_STEPS * style_sample_count) * 100)) + "%] "
            estimated_time = 0
            if steps != 0:
                estimated_time = (((end_time - start_time) / steps) * (
                        PROCESSING_STEPS * style_sample_count)) / 60
            _str += "(" + str(int(end_time - start_time)) + "s, estimated " + str(estimated_time) + "m)"
            print("", end="\x1b[2K\r")
            print(_str, end="\r", flush=True)

            steps += 1

        print("", end="\x1b[2K\r")
        print("Done! " + str(int(end_time - start_time)) + "s elapsed")

        input_image = tf.convert_to_tensor(image)
        keras.preprocessing.image.save_img("output.png", image[0])


load_style('vangog')
process_image(content_img, "styles")
plt.subplot(1, 2, 1)
plt.imshow(content_img)
plt.subplot(1, 2, 2)

plt.show()
