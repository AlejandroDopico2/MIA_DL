from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization


class AmazonDataset:
    TARGET_INDEX = 0
    DELAY = 2
    N_FEATURES = 6
    N_SAMPLES = 143

    def __init__(
        self,
        train_data: Tuple[List[str], List[int], int],
        test_data: Tuple[List[str], List[int]],
        max_features: Optional[int],
    ):

        X_train, y_train, sequence_length = train_data
        X_test, y_test = test_data

        self.sequence_length = sequence_length
        self.max_features = max_features

        X_train, X_test = self.transform(X_train, X_test)

        X_train, X_val, y_train, y_val = self.split(
            X_train, y_train, validation_split=0.2
        )

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = tf.convert_to_tensor(y_train)
        self.y_val = tf.convert_to_tensor(y_val)
        self.y_test = tf.convert_to_tensor(y_test)

    def transform(self, x_train, x_test):
        """
        Transforms text input to int input based on the vocabulary.

        Args:
        - x_train: Input training data.
        - x_test: Input test data.
        - y_train: Target training data.
        - y_test: Target test data.

        Returns:
        - x_train_int: Transformed integer input for training.
        - y_train: Converted tensor for training targets.
        - x_test_int: Transformed integer input for testing.
        - y_test: Converted tensor for testing targets.
        """

        # Define preprocessing layer
        precLayer = TextVectorization(
            max_tokens=self.max_features,
            standardize="lower_and_strip_punctuation",
            split="whitespace",
            output_mode="int",
            output_sequence_length=self.sequence_length,
        )

        # Adapt preprocessing layer on training data
        precLayer.adapt(x_train)

        # Transform training and test data
        x_train_int = precLayer(x_train)
        x_test_int = precLayer(x_test)

        if self.max_features is None:
            self.max_features = precLayer.vocabulary_size()

        return x_train_int, x_test_int

    def split(self, input_data, labels, validation_split=0.2):
        # Shuffle the indices
        num_samples = tf.shape(input_data)[0]
        indices = tf.range(num_samples)
        shuffled_indices = tf.random.shuffle(indices)

        # Define the split index
        split_index = tf.cast((1 - validation_split) * tf.cast(num_samples, tf.float32), tf.int32)


        # Split the data
        train_indices = shuffled_indices[:split_index]
        val_indices = shuffled_indices[split_index:]
        
        train_input = tf.gather(input_data, train_indices)
        val_input = tf.gather(input_data, val_indices)
        train_labels = tf.gather(labels, train_indices)
        val_labels = tf.gather(labels, val_indices)

        return train_input, val_input, train_labels, val_labels

    @classmethod
    def read_data(cls, lines: List[str]):
        textData = list()
        textLabel = list()
        lineLength = np.zeros(len(lines))

        for i, aLine in enumerate(lines):
            if not aLine:
                break
            label = aLine.split(" ")[0]
            lineLength[i] = len(aLine.split(" "))
            if label == "__label__1":
                textLabel.append(0)
                textData.append(aLine.removeprefix("__label__1 "))

            elif label == "__label__2":
                textLabel.append(1)
                textData.append(aLine.removeprefix("__label__2 "))

            else:
                print("\nError in load: ", i, aLine)
                exit()

        return textData, textLabel, int(np.average(lineLength) + 2 * np.std(lineLength))

    @classmethod
    def load(cls, train_path: str, test_path: str, max_features: Optional[int] = None):
        with open(train_path, "r", encoding="utf-8") as f:
            train_file_data = f.read()

        train_lines = train_file_data.split("\n")

        train_textData, train_textLabel, train_sequence_len = cls.read_data(train_lines)

        f.close()

        with open(test_path, "r", encoding="utf-8") as f:
            test_file_data = f.read()

        test_lines = test_file_data.split("\n")

        test_textData, test_textLabel, _ = cls.read_data(test_lines)

        f.close()

        return AmazonDataset(
            (train_textData, train_textLabel, train_sequence_len),
            (test_textData, test_textLabel),
            max_features=max_features,
        )
