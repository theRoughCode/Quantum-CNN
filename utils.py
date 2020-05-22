import cirq
import tensorflow as tf
import np
import sympy

def generate_random_circuit(qubits, symbol, depth):
    """Generate random circuit with the same structure from McClean et al."""
    circuit = cirq.Circuit()
    for qubit in qubits:
        circuit += cirq.ry(np.pi / 4.0)(qubit)

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

def random_quanv_circuit(qubits, symbol, connection_prob=0.1):
    num_qubits = len(qubits)
    # TODO: Implement arbitrary unitary rotation gate: http://www.vcpc.univie.ac.at/~ian/hotlist/qc/talks/bloch-sphere-rotations.pdf
    one_qubit_gate_set = [cirq.rx, cirq.ry, cirq.rz, cirq.Z**0.25, cirq.H]
    # TODO: Implement arbitrary unitary rotation gate
    # CNOT, SWAP, SqrtSWAP, Controlled U
    two_qubit_gate_set = [cirq.CNOT, cirq.SWAP, cirq.SWAP**0.5, cirq.CZ]

    circuit = cirq.Circuit()
    ops = []
    
    # connections[i][j] = True => has a 2 qubit gate between them
    connections = np.random.rand(num_qubits, num_qubits) <= connection_prob
    gates = np.random.randint(0, len(two_qubit_gate_set), (num_qubits, num_qubits))

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            # Add gate with prob
            if connections[i, j]:
                gate = two_qubit_gate_set[gates[i, j]]
                ops.append(gate.on(qubits[i], qubits[j]))
    
    # 2n^2 1-qubit gates
    num_one_qubit_gates = np.random.randint(0, 2*num_qubits)
    one_qubits_gates = tf.random.uniform((num_one_qubit_gates,), 0, len(one_qubit_gate_set), dtype=tf.int32)
    one_qubit_targets = tf.random.uniform((num_one_qubit_gates,), 0, num_qubits, dtype=tf.int32)
    for i in range(num_one_qubit_gates):
        gate = one_qubit_gate_set[one_qubits_gates[i]]
        target = qubits[one_qubit_targets[i]]
        if gate in [cirq.rx, cirq.ry, cirq.rz]:
          gate = gate(symbol)
        ops.append(gate.on(target))
    
    # Randomly shuffle operations
    np.random.shuffle(ops)

    for op in ops:
        circuit += op

    return circuit

