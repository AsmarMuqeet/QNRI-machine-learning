# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Estimator V2 implementation that uses the custom Aer_Sampler to compute expectation values."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers import DataBin, EstimatorPubLike, PrimitiveResult, PubResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.primitive_job import PrimitiveJob

from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library.standard_gates import HGate, SGate

# Use your sampler implementation
from AerSampler import Aer_Sampler


@dataclass
class Options:
    """Options for :class:`~.EstimatorV2`."""

    default_precision: float = 0.0
    """The default precision to use if none are specified in :meth:`~run`."""

    backend_options: dict = field(default_factory=dict)
    """backend_options: Options passed to AerSimulator via your Aer_Sampler."""

    run_options: dict = field(default_factory=dict)
    """run_options: Options passed to run (shots, seed_simulator, etc.)."""


class Aer_Estimator(BaseEstimatorV2):
    """Evaluates expectation values for provided quantum circuit and observable combinations.

    Uses *your* Aer_Sampler to sample bitstrings and turns those counts into expectation values.
    Optionally adds Gaussian noise ``N(expval, precision)`` when ``precision > 0``.
    """

    def __init__(self, *, options: dict | None = None):
        """
        Args:
            options: Controls the default precision (``default_precision``),
                     the backend options (``backend_options``), and
                     the runtime options (``run_options``).
        """
        self._options = Options(**options) if options else Options()

    @property
    def options(self) -> Options:
        """Return the options"""
        return self._options

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = self._options.default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _validate_pubs(self, pubs: list[EstimatorPub]):
        for i, pub in enumerate(pubs):
            if pub.precision < 0.0:
                raise ValueError(
                    f"The {i}-th pub has precision less than 0 ({pub.precision}). "
                    "But precision should be equal to or larger than 0."
                )

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs], metadata={"version": 2})

    # ---------- helpers ----------

    @staticmethod
    def _basis_change_for_pauli(circ: QuantumCircuit, pauli: str) -> QuantumCircuit:
        """
        Return a new circuit with basis rotations so measuring in Z yields eigenvalues of `pauli`.
        X -> H, Y -> S^† then H, Z/I -> no change. Length is padded with I to num_qubits.
        """
        new_circ = circ.copy()
        n = new_circ.num_qubits
        pauli = pauli.ljust(n, "I")
        for q, p in enumerate(pauli):
            if p == "X":
                new_circ.append(HGate(), [q])
            elif p == "Y":
                new_circ.append(SGate().inverse(), [q])
                new_circ.append(HGate(), [q])
        return new_circ

    @staticmethod
    def _expval_from_counts(counts: dict[str, int], pauli: str, n_qubits: int) -> float:
        """
        Compute <P> from computational-basis counts after basis change.
        Eigenvalue = (-1)^{parity of '1's over non-I positions}.
        """
        active = [i for i, p in enumerate(pauli.ljust(n_qubits, "I")) if p != "I"]
        total = 0
        shots = 0
        for key, freq in counts.items():
            s = key.decode() if isinstance(key, bytes) else str(key)
            s = s.replace(" ", "")
            if s.startswith("0x"):
                s = format(int(s, 16), f"0{n_qubits}b")
            # Rightmost bit corresponds to qubit 0 (due to our measure mapping)
            parity = 0
            for q in active:
                bit = s[-(q + 1)] if q < len(s) else "0"
                if bit == "1":
                    parity ^= 1
            ev = -1.0 if parity == 1 else 1.0
            total += ev * int(freq)
            shots += int(freq)
        return 0.0 if shots == 0 else total / shots

    # ---------- main per-pub execution ----------

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision

        # Broadcasting bookkeeping (kept for output shape compatibility)
        param_shape = parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, observables)
        flat_indices = list(param_indices.ravel())

        # Collect unique Pauli strings across all observable entries
        paulis = {pauli for obs_dict in observables.ravel() for pauli in obs_dict.keys()}
        n_qubits = circuit.num_qubits

        # Build one measurement circuit per unique Pauli
        pauli_circs = []
        pauli_list = []
        for P in sorted(paulis):
            meas_circ = self._basis_change_for_pauli(circuit, P)

            # Always measure all qubits into a fresh creg 'c' of size n_qubits
            creg = ClassicalRegister(n_qubits, "c")
            meas_circ.add_register(creg)
            # Map qubit i -> cbit i so that rightmost bit is qubit 0 in bitstrings
            meas_circ.measure(meas_circ.qubits, list(creg))

            pauli_circs.append(meas_circ)
            pauli_list.append(P)

        # Configure your sampler
        default_shots = self.options.run_options.get("shots", 1024)
        seed = self.options.run_options.get("seed_simulator")
        sampler = Aer_Sampler(
            default_shots=default_shots,
            seed=seed,
            options={
                "backend_options": self.options.backend_options,
                # pass through any remaining run_options (excluding shots/seed which we set above)
                "run_options": {
                    k: v for k, v in self.options.run_options.items() if k not in ("shots", "seed_simulator")
                },
            },
        )

        # Build pubs for the sampler.
        # NOTE: your Aer_Sampler binds *the first* parameter set if broadcasting is provided.
        # To support full broadcasting, expand here: one pub per (param_set, Pauli) and then
        # reshape at the end.
        pubs = [(circ, parameter_values, default_shots) for circ in pauli_circs]

        job = sampler.run(pubs)
        samp_res = job.result()  # aligns with pauli_list

        print(samp_res)

        # P -> <P> (for the selected/first parameter set)
        pauli_to_ev: dict[str, float] = {}
        for i, P in enumerate(pauli_list):
            counts = samp_res[i].data.c.get_counts()
            print(counts)
            pauli_to_ev[P] = self._expval_from_counts(counts, P, n_qubits)

        # Assemble evs/stds with broadcasting shape (reuse first parameter set for all entries)
        evs = np.zeros_like(bc_param_ind, dtype=float)
        stds = np.full(bc_param_ind.shape, precision)
        for index in np.ndindex(*bc_param_ind.shape):
            total = 0.0
            for P, coeff in bc_obs[index].items():
                total += pauli_to_ev[P] * coeff
            evs[index] = total

        # Optional Gaussian noise
        if precision > 0:
            rng = np.random.default_rng(seed)
            if not np.all(np.isreal(evs)):
                raise ValueError("Given operator is not Hermitian and noise cannot be added.")
            evs = rng.normal(evs, precision, evs.shape)

        return PubResult(
            DataBin(evs=evs, stds=stds, shape=evs.shape),
            metadata={
                "target_precision": precision,
                "circuit_metadata": pub.circuit.metadata,
                "simulator_metadata": {"engine": "sampler"},
            },
        )
