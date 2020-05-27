import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import collections
import numpy as np

from gates import ZXGate
from utils import hinge_accuracy, binary_accuracy

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
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    # TODO: Experiment with 3 alternating layers of ZX/XX as used by Farhi et al.
    for i in range(num_layers):
        builder.add_layer(circuit, ZXGate, "zz1{}".format(i + 1))
        builder.add_layer(circuit, cirq.XX, "xx{}".format(i + 1))

    return circuit, cirq.Z(readout)

class QNN():
    def __init__(self, loss=tf.keras.losses.BinaryCrossentropy, optimizer=tf.keras.optimizers.Adam,
                 lr=0.02, metrics=[binary_accuracy]):
        self.loss = loss()
        self.optimizer = optimizer(learning_rate=lr)
        self.metrics = metrics
        self.model = self.build_model()

    def build_model(self):
        model_circuit, model_readout = create_quantum_model()

        # The input is the data-circuit, encoded as a tf.string
        q_img = tf.keras.layers.Input(shape=(), dtype=tf.string)
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        x = tfq.layers.PQC(model_circuit, model_readout)(q_img)
        # Map output from [-1, 1] to [0, 1]
        out = (x + 1) / 2

        # Build the Keras model.
        model = tf.keras.Model(inputs=q_img, outputs=out)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return model
