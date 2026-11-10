"""Explicit field models; integer labels in extensions are NOT residues mod q."""

from functools import lru_cache

import galois


class Field:
    def __init__(self, order: int, irreducible_poly=None):
        if type(order) is not int or order < 2:
            raise ValueError("field order must be an integer prime power >= 2")
        # The library validates prime-power order and any supplied polynomial.
        self.GF = galois.GF(order, irreducible_poly=irreducible_poly,
                            compile="python-calculate", verify=True)
        self.q = order
        self.p = int(self.GF.characteristic)
        self.degree = int(self.GF.degree)
        self._add = self._mul = None
        if self.degree > 1 and order <= 64:
            elements = self.GF.Range(0, order)
            self._add = (elements[:, None] + elements[None, :]).tolist()
            self._mul = (elements[:, None] * elements[None, :]).tolist()

    def element(self, value):
        if type(value) is not int or not 0 <= value < self.q:
            raise ValueError(f"entries must be integer field labels 0 <= x < {self.q}")
        return value

    def add(self, a, b):
        if self.degree == 1:
            return (a + b) % self.p
        if self._add is not None:
            return self._add[a][b]
        return int(self.GF(a) + self.GF(b))

    def neg(self, a):
        return (-a) % self.p if self.degree == 1 else int(-self.GF(a))

    def sub(self, a, b):
        return self.add(a, self.neg(b))

    def mul(self, a, b):
        if self.degree == 1:
            return (a * b) % self.p
        if self._mul is not None:
            return self._mul[a][b]
        return int(self.GF(a) * self.GF(b))

    @lru_cache(maxsize=4096)
    def inv(self, a):
        if a == 0:
            raise ZeroDivisionError("zero has no multiplicative inverse")
        return pow(a, -1, self.p) if self.degree == 1 else int(self.GF(a) ** -1)

    def metadata(self):
        return {
            "order": self.q,
            "characteristic": self.p,
            "degree": self.degree,
            "irreducible_poly": self.GF.irreducible_poly.coeffs.tolist()
                if self.degree > 1 else None,
            "encoding": "polynomial-basis integer labels; coefficients are base-p digits",
        }
