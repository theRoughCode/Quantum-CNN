import tensorflow_quantum as tfq
import tensorflow as tf
import numpy as np
import collections
import cirq

X_FLIP = 'X_FLIP'
Y_ROT = 'Y_ROT'
Y_POW = 'Y_POW'

def get_preprocessor(name):
    if name == X_FLIP:
        return preprocess_x_flip
    if name == Y_ROT:
        return preprocess_y_rot
    if name == Y_POW:
        return preprocess_y_pow

# Remove contradicting images (multiple labels)
def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
        mapping[tuple(x.flatten())].add(y)
        
    new_x = []
    new_y = []
    for x,y in zip(xs, ys):
        labels = mapping[tuple(x.flatten())]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(list(labels)[0])
        else:
            # Throw out images that match more than one label.
            pass
    
    return np.array(new_x), np.array(new_y)

def get_mnist(img_dim=4, preprocessor_name=X_FLIP, num_examples=500):
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    def filter_36(x, y):
        keep = (y == 3) | (y == 6)
        x, y = x[keep], y[keep]
        y = y == 3
        return x, y

    x_train, y_train = filter_36(x_train, y_train)
    x_test, y_test = filter_36(x_test, y_test)

    if img_dim is not None:
        # Scale images
        x_train = tf.image.resize(x_train, (img_dim, img_dim)).numpy()
        x_test = tf.image.resize(x_test, (img_dim, img_dim)).numpy()

    # Remove contradicting images
    x_train, y_train = remove_contradicting(x_train, y_train)

    x_train = x_train[:num_examples]
    y_train = y_train[:num_examples]

    preprocessor = get_preprocessor(preprocessor_name)

    x_train_circ = [preprocessor(x, img_dim) for x in x_train]
    x_test_circ = [preprocessor(x, img_dim) for x in x_test]
    # Convert to tensor
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

    return (x_train_tfcirc, y_train), (x_test_tfcirc, y_test)

def preprocess_x_flip(image, dims=4, threshold=0.5):
    """Encode truncated classical image by applying X if above threshold."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(dims, dims)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
      if value > threshold:
        circuit.append(cirq.X(qubits[i]))
    return circuit

def preprocess_y_rot(image, dims=4):
    """Encode truncated classical image by rotating through a Y gate based
       on pixel value."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(dims, dims)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
      circuit.append(cirq.ry(np.pi * value)(qubits[i]))
    return circuit

def preprocess_y_pow(image, dims=4):
    """Encode truncated classical image by applying a Y gate and weighting
       by pixel value."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(dims, dims)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
      circuit.append(cirq.Y(qubits[i]) ** (np.pi * value))
    return circuit