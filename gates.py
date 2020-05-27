import cirq
import numpy as np

class UGate():
    """
    Unitary rotation gate about arbitrary axis (phi, theta)
    with angle alpha.
    Rn(α) = Rz(φ)Ry(θ)Rz(α)Ry(−θ)Rz(−φ)
    Refer to: http://www.vcpc.univie.ac.at/~ian/hotlist/qc/talks/bloch-sphere-rotations.pdf
    """
    def __init__(self, phi, theta, alpha, qubit=None):
        self.phi = phi
        self.theta = theta
        self.alpha = alpha
        self.qubit = qubit
        self.exponent = 1
        self.gates  = [
            cirq.rz(phi)**-1, cirq.ry(theta)**-1,
            cirq.rz(alpha),
            cirq.ry(theta), cirq.rz(phi)
        ]

    def __call__(self, qubit):
        return self.on(qubit)

    def on(self, qubit):
        return UGate(self.phi, self.theta, self.alpha, qubit)
    
    def __pow__(self, symbol):
        self.exponent = symbol
        return [gate.on(self.qubit)**symbol for gate in self.gates]

    def __str__(self):
        if self.exponent == 1:
            return 'U({})'.format(self.alpha)
        return 'U({})**{!r}'.format(self.alpha, self.exponent)

class ControlledUGate():
    """
    Controlled U Gate using RZ and CNOT.
    See: https://qiskit.org/textbook/ch-gates/more-circuit-identities.html
    """
    def __init__(self, theta, qubit=None, readout=None):
        self.theta = theta
        self.qubit = qubit
        self.readout = readout
        self.exponent = 1

    def __call__(self, qubit, readout):
        return self.on(qubit, readout)

    def on(self, qubit, readout):
        return ControlledUGate(self.theta, qubit, readout)
    
    def __pow__(self, symbol):
        self.exponent = symbol
        return [(cirq.rz(self.theta)**-1)(self.readout)**symbol,
                 cirq.CNOT(self.qubit, self.readout)**symbol,
                 cirq.rz(self.theta)(self.readout)**symbol]

    def __str__(self):
        if self.exponent == 1:
            return 'CU({})'.format(self.theta)
        return 'CU({})**{!r}'.format(self.theta, self.exponent)

class ZXGate():
    def __init__(self, qubit=None, readout=None, exponent=1):
        self.qubit = qubit
        self.readout = readout
        self.exponent = exponent
    
    def __call__(self, qubit, readout):
        return self.on(qubit, readout)

    def on(self, qubit, readout):
        return ZXGate(qubit, readout)
    
    def __pow__(self, symbol):
        self.exponent = symbol
        return [cirq.H(self.readout)**symbol, cirq.ZZ(self.qubit, self.readout)**symbol, cirq.H(self.readout)**symbol]
    
    def __str__(self):
        if self.exponent == 1:
            return 'ZX'
        return 'ZX**{!r}'.format(self.exponent)
