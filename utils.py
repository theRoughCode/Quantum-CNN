import cirq
import tensorflow as tf
import numpy as np
import sympy

from gates import UGate, ControlledUGate

@tf.function
def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

@tf.function
def binary_accuracy(y_true, y_pred):
  y_true = tf.squeeze(y_true)
  y_pred = tf.squeeze(y_pred)
  return tf.math.reduce_mean(tf.cast(tf.math.equal(y_true, tf.round(y_pred)), tf.float32))

def generate_random_circuit(qubits, symbol, depth):
    """Generate random circuit with the same structure from McClean et al."""
    circuit = cirq.Circuit()
    for qubit in qubits:
        circuit += (cirq.H**0.5)(qubit)

    for d in range(depth):
        # Add a series of single qubit rotations.
        for i, qubit in enumerate(qubits):
            random_n = np.random.uniform()
            random_rot = np.random.uniform(high=2.0 * np.pi) if i != 0 or d != 0 else symbol
            if random_n > 2. / 3.:
                # Add a Z.
                circuit += cirq.rz(random_rot)(qubit)
            elif random_n > 1. / 3.:
                # Add a Y.
                circuit += cirq.ry(random_rot)(qubit)
            else:
                # Add a X.
                circuit += cirq.rx(random_rot)(qubit)

        # Add CZ ladder.
        for src, dest in zip(qubits, qubits[1:]):
            circuit += cirq.CZ(src, dest)

    return circuit

def random_circuit_pl(qubits, prefix, ratio_imprim=0.3, seed=42):
    np.random.seed(seed)
    rotations = [cirq.rx, cirq.ry, cirq.rz]
    circuit = cirq.Circuit()

    i = 0
    while i < len(qubits):
        if np.random.random() > ratio_imprim:
            # Apply a random rotation gate to a random wire
            gate = np.random.choice(rotations)
            qubit = np.random.choice(qubits)
            symbol = sympy.Symbol(prefix + str(i))
            circuit.append(gate(symbol).on(qubit))
            i += 1
        else:
            targets = np.random.choice(qubits, (2), replace=False)
            circuit.append(cirq.CNOT(targets[0], targets[1]))
    return circuit

def random_quanv_circuit(qubits, prefix, connection_prob=0.3):
    num_qubits = len(qubits)
    # NOTE: Phase gate not included
    one_qubit_gate_set = [cirq.rx, cirq.ry, cirq.rz, cirq.Z**0.25, cirq.H, UGate]
    # CNOT, SWAP, SqrtSWAP, Controlled U
    two_qubit_gate_set = [cirq.CNOT, cirq.SWAP, cirq.SWAP**0.5, ControlledUGate]

    circuit = cirq.Circuit()
    ops = []
    
    # connections[i][j] = True => has a 2 qubit gate between them
    connections = np.random.rand(num_qubits, num_qubits) <= connection_prob
    gates = np.random.choice(two_qubit_gate_set, (num_qubits, num_qubits))
    angles = np.random.uniform(high=2 * np.pi, size=(num_qubits, num_qubits))
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            # Add gate with prob
            if connections[i, j]:
                symbol = sympy.Symbol("{}_{}_{}".format(prefix, i, j))
                angle = angles[i, j]
                gate = gates[i, j]
                if gate == ControlledUGate:
                    gate = gate(angle)
                targets = np.random.choice([qubits[i], qubits[j]], (2), replace=False)
                ops.append(gate.on(targets[0], targets[1])**symbol)
    
    # 2n^2 1-qubit gates
    num_one_qubit_gates = np.random.randint(0, 2*num_qubits)
    gates = np.random.choice(one_qubit_gate_set, num_one_qubit_gates)
    targets = np.random.choice(qubits, num_one_qubit_gates)
    angles = np.random.uniform(high=2 * np.pi, size=num_one_qubit_gates)
    for i, (gate, target, angle) in enumerate(zip(gates, targets, angles)):
        if gate in [cirq.rx, cirq.ry, cirq.rz]:
            gate = gate(angle)
        elif gate == UGate:
            phi, theta = np.random.uniform(0, 2 * np.pi, 2)
            gate = gate(phi, theta, angle)
        symbol = sympy.Symbol("{}_{}".format(prefix, i))
        ops.append(gate.on(target)**symbol)
    
    # Randomly shuffle operations
    np.random.shuffle(ops)
    circuit.append(ops)

    return circuit

