import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy

def one_qubit_unitary(bit, symbols):
    """Make a Cirq circuit enacting a rotation of the bloch sphere about the X,
    Y and Z axis, that depends on the values in `symbols`.
    Bit: 1
    Symbols: 3
    """
    return cirq.Circuit(
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2])


def two_qubit_unitary(bits, symbols):
    """Make a Cirq circuit that creates an arbitrary two qubit unitary.
    Bits: 2
    Symbols: 15
    """
    # Corollary 6 of https://arxiv.org/pdf/quant-ph/0507171.pdf
    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(bits[0], symbols[0:3])
    circuit += one_qubit_unitary(bits[1], symbols[3:6])
    circuit += [cirq.ZZ(*bits)**symbols[7]]
    circuit += [cirq.YY(*bits)**symbols[8]]
    circuit += [cirq.XX(*bits)**symbols[9]]
    circuit += one_qubit_unitary(bits[0], symbols[9:12])
    circuit += one_qubit_unitary(bits[1], symbols[12:])
    return circuit


def two_qubit_pool(source_qubit, sink_qubit, symbols):
    """Make a Cirq circuit to do a parameterized 'pooling' operation, which
    attempts to reduce entanglement down from two qubits to just one.
    Source: 1
    Sink: 1
    Symbols: 6
    """
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, symbols[0:3])
    source_basis_selector = one_qubit_unitary(source_qubit, symbols[3:6])
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    pool_circuit.append(sink_basis_selector**-1)
    return pool_circuit

def quantum_conv_circuit(bits, symbols):
    """Quantum Convolution Layer following the above diagram.
    Return a Cirq circuit with the cascade of `two_qubit_unitary` applied
    to all pairs of qubits in `bits`.
    Bits: N
    Symbols: 15
    """
    # TODO: Use different symbols?
    circuit = cirq.Circuit()
    for first, second in zip(bits[0::2], bits[1::2]):
        circuit += two_qubit_unitary([first, second], symbols)
    for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
        circuit += two_qubit_unitary([first, second], symbols)
    return circuit

def quantum_pool_circuit(source_bits, sink_bits, symbols):
    """A layer that specifies a quantum pooling operation.
    A Quantum pool tries to learn to pool the relevant information from two
    qubits onto 1.
    Source: N
    Sink: N
    Symbols: 6
    """
    circuit = cirq.Circuit()
    for source, sink in zip(source_bits, sink_bits):
        circuit += two_qubit_pool(source, sink, symbols)
    return circuit

def multi_readout_model_circuit(qubits):
    """
    Make a model circuit with 1 quantum pool and conv operations.
    Reduces N qubits to N/2 qubits.
    """
    num_qubits = len(qubits)
    mid = num_qubits // 2
    model_circuit = cirq.Circuit()
    symbols = sympy.symbols('qconv0:21')
    model_circuit += quantum_conv_circuit(qubits, symbols[0:15])
    model_circuit += quantum_pool_circuit(qubits[:mid], qubits[mid:],
                                          symbols[15:21])
    return model_circuit

class QCNN(object):
    def __init__(self, img_dim=4, lr=0.02, num_filters=8):
        self.img_dim = img_dim
        self.lr = lr
        self.num_filters = num_filters
        self.model = self.build_model()
    
    def build_model(self):
        qubits = cirq.GridQubit.rect(self.img_dim, self.img_dim)
        readouts = [cirq.Z(bit) for bit in qubits[len(qubits) // 2:]]
        inp = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        x = tfq.layers.PQC(multi_readout_model_circuit(qubits),
                           readouts)(inp)

        x = tf.keras.layers.Dense(self.num_filters)(x)
        x = tf.keras.layers.Dropout(0.125)(x)
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=[inp], outputs=[out])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model
