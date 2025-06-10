import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, metrics
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.layers import Conv2D, Activation, Input, Add, MaxPooling2D, Dense, Dropout, BatchNormalization, \
    Concatenate, Lambda, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from pyprojroot.here import here
import matplotlib.pyplot as plt


# Define scaling function for residual connections
def scaling(x, scale):
    return x * scale


# Define the InceptionResNetV2 model
def InceptionResNetV2():
    inputs = Input(shape=(160, 160, 3))
    x = Conv2D(32, 3, strides=2, padding='valid', use_bias=False, name='Conv2d_1a_3x3')(inputs)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_1a_3x3_BatchNorm')(x)
    x = Activation('relu', name='Conv2d_1a_3x3_Activation')(x)
    x = Conv2D(32, 3, strides=1, padding='valid', use_bias=False, name='Conv2d_2a_3x3')(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2a_3x3_BatchNorm')(x)
    x = Activation('relu', name='Conv2d_2a_3x3_Activation')(x)
    x = Conv2D(64, 3, strides=1, padding='same', use_bias=False, name='Conv2d_2b_3x3')(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2b_3x3_BatchNorm')(x)
    x = Activation('relu', name='Conv2d_2b_3x3_Activation')(x)
    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
    x = Conv2D(80, 1, strides=1, padding='valid', use_bias=False, name='Conv2d_3b_1x1')(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_3b_1x1_BatchNorm')(x)
    x = Activation('relu', name='Conv2d_3b_1x1_Activation')(x)
    x = Conv2D(192, 3, strides=1, padding='valid', use_bias=False, name='Conv2d_4a_3x3')(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4a_3x3_BatchNorm')(x)
    x = Activation('relu', name='Conv2d_4a_3x3_Activation')(x)
    x = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name='Conv2d_4b_3x3')(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4b_3x3_BatchNorm')(x)
    x = Activation('relu', name='Conv2d_4b_3x3_Activation')(x)

    # 5x Block35 (Inception-ResNet-A)
    for i in range(1, 6):
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name=f'Block35_{i}_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block35_{i}_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name=f'Block35_{i}_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name=f'Block35_{i}_Branch_1_Conv2d_0a_1x1')(
            x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block35_{i}_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name=f'Block35_{i}_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name=f'Block35_{i}_Branch_1_Conv2d_0b_3x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block35_{i}_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name=f'Block35_{i}_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name=f'Block35_{i}_Branch_2_Conv2d_0a_1x1')(
            x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block35_{i}_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name=f'Block35_{i}_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name=f'Block35_{i}_Branch_2_Conv2d_0b_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block35_{i}_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name=f'Block35_{i}_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name=f'Block35_{i}_Branch_2_Conv2d_0c_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block35_{i}_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name=f'Block35_{i}_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name=f'Block35_{i}_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name=f'Block35_{i}_Conv2d_1x1')(mixed)
        up = Lambda(scaling, arguments={'scale': 0.17})(up)
        x = Add()([x, up])
        x = Activation('relu', name=f'Block35_{i}_Activation')(x)

    # Mixed 6a (Reduction-A)
    branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name='Mixed_6a_Branch_0_Conv2d_1a_3x3')(x)
    branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                  name='Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
    branch_0 = Activation('relu', name='Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
    branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Mixed_6a_Branch_1_Conv2d_0a_1x1')(x)
    branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                  name='Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
    branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
    branch_1 = Conv2D(192, 3, strides=1, padding='same', use_bias=False, name='Mixed_6a_Branch_1_Conv2d_0b_3x3')(
        branch_1)
    branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                  name='Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
    branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
    branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name='Mixed_6a_Branch_1_Conv2d_1a_3x3')(
        branch_1)
    branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                  name='Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
    branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
    branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_6a_Branch_2_MaxPool_1a_3x3')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=3, name='Mixed_6a')(branches)

    # 10x Block17 (Inception-ResNet-B)
    for i in range(1, 11):
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name=f'Block17_{i}_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block17_{i}_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name=f'Block17_{i}_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False,
                          name=f'Block17_{i}_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block17_{i}_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name=f'Block17_{i}_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name=f'Block17_{i}_Branch_1_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block17_{i}_Branch_1_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name=f'Block17_{i}_Branch_1_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name=f'Block17_{i}_Branch_1_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block17_{i}_Branch_1_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name=f'Block17_{i}_Branch_1_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name=f'Block17_{i}_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name=f'Block17_{i}_Conv2d_1x1')(mixed)
        up = Lambda(scaling, arguments={'scale': 0.1})(up)
        x = Add()([x, up])
        x = Activation('relu', name=f'Block17_{i}_Activation')(x)

    # Mixed 7a (Reduction-B)
    branch_0 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name='Mixed_7a_Branch_0_Conv2d_0a_1x1')(x)
    branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                  name='Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm')(branch_0)
    branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation')(branch_0)
    branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name='Mixed_7a_Branch_0_Conv2d_1a_3x3')(
        branch_0)
    branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                  name='Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
    branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
    branch_1 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name='Mixed_7a_Branch_1_Conv2d_0a_1x1')(x)
    branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                  name='Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
    branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
    branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name='Mixed_7a_Branch_1_Conv2d_1a_3x3')(
        branch_1)
    branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                  name='Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
    branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
    branch_2 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name='Mixed_7a_Branch_2_Conv2d_0a_1x1')(x)
    branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                  name='Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
    branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
    branch_2 = Conv2D(256, 3, strides=1, padding='same', use_bias=False, name='Mixed_7a_Branch_2_Conv2d_0b_3x3')(
        branch_2)
    branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                  name='Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
    branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
    branch_2 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name='Mixed_7a_Branch_2_Conv2d_1a_3x3')(
        branch_2)
    branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                  name='Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm')(branch_2)
    branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation')(branch_2)
    branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_7a_Branch_3_MaxPool_1a_3x3')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=3, name='Mixed_7a')(branches)

    # 6x Block8 (Inception-ResNet-C)
    for i in range(1, 7):
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name=f'Block8_{i}_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block8_{i}_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name=f'Block8_{i}_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name=f'Block8_{i}_Branch_1_Conv2d_0a_1x1')(
            x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block8_{i}_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name=f'Block8_{i}_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False,
                          name=f'Block8_{i}_Branch_1_Conv2d_0b_1x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block8_{i}_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name=f'Block8_{i}_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False,
                          name=f'Block8_{i}_Branch_1_Conv2d_0c_3x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=f'Block8_{i}_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name=f'Block8_{i}_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name=f'Block8_{i}_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name=f'Block8_{i}_Conv2d_1x1')(mixed)
        up = Lambda(scaling, arguments={'scale': 0.2 if i < 6 else 1})(up)
        x = Add()([x, up])
        if i < 6:
            x = Activation('relu', name=f'Block8_{i}_Activation')(x)

    # Classification block
    x = GlobalAveragePooling2D(name='AvgPool')(x)
    x = Dropout(1.0 - 0.8, name='Dropout')(x)
    x = Dense(128, use_bias=False, name='Bottleneck')(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm')(x)

    # Create model
    model = Model(inputs, x, name='inception_resnet_v1')
    model.load_weights(here("/kaggle/input/weightssss/facenet_keras_weights.h5"))

    # Freeze layers up to Mixed_7a, unfreeze Block8 and beyond
    for layer in model.layers:
        if layer.name.startswith('Block8') or layer.name.startswith('Mixed_7a') or layer.name in ['AvgPool', 'Dropout',
                                                                                                  'Bottleneck',
                                                                                                  'Bottleneck_BatchNorm']:
            layer.trainable = True
        else:
            layer.trainable = False

    return model


# Generate triplets
def generate_triplets(path, num_triplets_each_person=500):
    anchors, positives, negatives = [], [], []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path) or len(os.listdir(folder_path)) < 2:
            continue
        images = os.listdir(folder_path)
        for _ in range(num_triplets_each_person):
            anchor = random.choice(images)
            positive = random.choice(images)
            other_folder = random.choice(
                [f for f in os.listdir(path) if f != folder and os.path.isdir(os.path.join(path, f))])
            negative = random.choice(os.listdir(os.path.join(path, other_folder)))
            anchors.append(os.path.join(folder_path, anchor))
            positives.append(os.path.join(folder_path, positive))
            negatives.append(os.path.join(path, other_folder, negative))
    return anchors, positives, negatives


# Preprocess images
def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [160, 160])
    return image


def preprocess_triplets(anchor, positive, negative):
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


# Prepare dataset
def prepare_data(train_size=5000):
    anchor, positive, negative = generate_triplets(here("/kaggle/input/dataset26/dataset"), train_size // 10)
    dataset = tf.data.Dataset.from_tensor_slices((anchor, positive, negative))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


# Visualize triplets
def visualize(anchor, positive, negative):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(anchor)
    plt.title("Anchor")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(positive)
    plt.title("Positive")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(negative)
    plt.title("Negative")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Distance Layer
class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


# FaceNet Model with Triplet Loss
class FaceNetModel(Model):
    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]


# Build Siamese Network
def for_training_model(target_shape=(160, 160)):
    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))
    embedding = InceptionResNetV2()

    distances = DistanceLayer()(
        embedding(preprocess_input(anchor_input)),
        embedding(preprocess_input(positive_input)),
        embedding(preprocess_input(negative_input)),
    )

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
    return model, embedding


# Training Process
def training_process(epochs, batch_size, learning_rate, margin, train_size, cache):
    dataset = prepare_data(train_size)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if cache:
        dataset = dataset.cache()

    for anchor, positive, negative in dataset.take(1):
        visualize(anchor[0].numpy(), positive[0].numpy(), negative[0].numpy())

    model_triple_loss, embedding_model = for_training_model()
    model_triple_loss = FaceNetModel(model_triple_loss, margin)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model_triple_loss.compile(optimizer=optimizer)

    model_triple_loss.fit(dataset, epochs=epochs)

    os.makedirs(here("/kaggle/working/fine_tune_model"), exist_ok=True)
    embedding_model.save(here("/kaggle/working/fine_tune_model/embedding_model_final.h5"))


# Configuration
def get_config():
    return {
        'epochs': 130,
        'batch_size': 32,
        'learning_rate': 1e-6,
        'margin': 0.5,
        'train_size': 5000,
        'cache': True
    }


if __name__ == "__main__":
    config = get_config()
    training_process(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        margin=config['margin'],
        train_size=config['train_size'],
        cache=config['cache']
    )