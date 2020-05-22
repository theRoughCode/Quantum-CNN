import cirq
import numpy as np

class ZXGate(cirq.ops.eigen_gate.EigenGate, cirq.ops.gate_features.TwoQubitGate):
  """
  ZXGate with variable weight.
  Retrieved from Quantum Computing: An Applied Approach by Jack D. Hidary. Pg 156
  Link: https://books.google.ca/books?id=nymsDwAAQBAJ&pg=PA156&lpg=PA156&dq=cirq+ZX+gate&source=bl&ots=8-budtYoQv&sig=ACfU3U3bNl9oWz1NEZvrqKw1l_0YGHmZyA&hl=en&sa=X&ved=2ahUKEwid3bqql8TpAhWIUt8KHQYqDnUQ6AEwAXoECAoQAQ#v=onepage&q=cirq%20ZX%20gate&f=false
  """

  def __init__(self, weight=1):
    self.weight = weight
    super().__init__(exponent=weight)

  def _eigen_components(self):
    return [
            (1, np.array([[0.5, 0.5, 0, 0],
                          [0.5, 0.5, 0, 0],
                          [0, 0, 0.5, -0.5],
                          [0, 0, -0.5, 0.5]])),
            (-1, np.array([[0.5, -0.5, 0, 0],
                           [-0.5, 0.5, 0, 0],
                           [0, 0, 0.5, 0.5],
                           [0, 0, 0.5, 0.5]]))
    ]

  # Lets weight be a Symbol. Useful for parameterization.
  def _resolve_parameters_(self, param_resolver):
    return ZXGate(weight=param_resolver.value_of(self.weight))

  def _circuit_diagram_info_(self, args):
    return cirq.protocols.CircuitDiagramInfo(
        wire_symbols=('Z', 'X'),
        exponent=self.weight
    )

  def __str__(self) -> str:
    if self._global_shift == -0.5:
      if self._exponent == 1:
          return 'ZX(π)'
      return f'ZX({self._exponent!r}π)'
    if self.exponent == 1:
      return 'ZX'
    return f'ZX**{self._exponent!r}'

  def __repr__(self) -> str:
    if self._exponent == 1:
      return 'cirq.ZX'
    return f'(cirq.ZX**{cirq._compat.proper_repr(self._exponent)})'