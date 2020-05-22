import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import collections
import numpy as np

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

def create_quantum_model(num_layers=1):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()
    
    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    # TODO: experiment with ZX gate instead of ZZ (as in https://arxiv.org/pdf/1802.06002.pdf)
    # Farhi et al. used 3 alternating layers of ZX/XX
    for i in range(num_layers):
        builder.add_layer(circuit, cirq.XX, "xx{}".format(i + 1))
        builder.add_layer(circuit, cirq.ZZ, "zz1{}".format(i + 1))

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)

class QNN(object):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model_circuit, model_readout = create_quantum_model()
        # Build the Keras model.
        model = tf.keras.Sequential([
            # The input is the data-circuit, encoded as a tf.string
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            # The PQC layer returns the expected value of the readout gate, range [-1,1].
            tfq.layers.PQC(model_circuit, model_readout),
        ])
        model.compile(loss=tf.keras.losses.Hinge(), optimizer=tf.keras.optimizers.Adam(),
                      metrics=[hinge_accuracy])
        return model
