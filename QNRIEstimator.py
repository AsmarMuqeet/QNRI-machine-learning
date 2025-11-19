from __future__ import annotations
from dataclasses import dataclass, field

from collections.abc import Iterable

import numpy as np

from qiskit.quantum_info import SparsePauliOp

from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers import DataBin, EstimatorPubLike, PrimitiveResult, PubResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives import PrimitiveJob
from qiskit_aer import AerSimulator
from qiskit import generate_preset_pass_manager
from qaas import QBackend, QProvider
from py4lexis.session import LexisSession

@dataclass
class Options:
    """Options for :class:`~.SamplerV2`."""

    backend_options: dict = field(default_factory=dict)
    """backend_options: Options passed to AerSimulator."""

    run_options: dict = field(default_factory=dict)
    """run_options: Options passed to run."""

class Aer_Estimator(BaseEstimatorV2):

    def __init__(
        self, *, shots: int = 1024, seed: np.random.Generator | int | None = None, options: dict | None = None
    ):
        """
        Args:
            default_precision: The default precision for the estimator if not specified during run.
            seed: The seed or Generator object for random number generation.
                If None, a random seeded default RNG will be used.
        """
        self._seed = seed
        self._options = Options(**options) if options else Options()
        self._backend = AerSimulator(**self._options.backend_options)
        self.pass_manager = generate_preset_pass_manager(backend=self._backend, optimization_level=3,seed_transpiler=self.seed)
        self._shots = shots
        self._rng = np.random.default_rng(seed)


    @property
    def seed(self) -> np.random.Generator | int | None:
        """Return the seed or Generator object for random number generation."""
        return self._seed

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = 0
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]

        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs], metadata={"version": 2})

    def _measure_and_get_counts(self, circuit):
            """Measure all qubits in Z basis and return counts."""
            meas_circ = circuit.copy()
            meas_circ.measure_all()

            # adjust run_options not to overwrite existing options
            run_options = self._options.run_options.copy()
            for key in ["shots", "parameter_binds", "memory"]:
                if key in run_options:
                    del run_options[key]
            if self._seed is not None and "seed_simulator" in run_options:
                del run_options["seed_simulator"]

            transpiled_circuits = self.pass_manager.run(meas_circ)
            job = self._backend.run(transpiled_circuits,
                                    shots=self._shots,
                                    **run_options)
            result = job.result()
            counts = result.get_counts()
            return counts

    # --- basis change for a single Pauli label -------------------------------
    @staticmethod
    def _apply_basis_change_for_label(circuit, label: str):
        """
        Apply basis-change gates so that measuring in Z basis
        gives eigenvalues of the Pauli string `label`.

        Assumes `label[i]` acts on qubit i (Qiskit convention).
        X -> H
        Y -> Sdg + H
        Z/I -> nothing
        """
        for qubit, p in enumerate(label):
            if p == "X":
                circuit.h(qubit)
            elif p == "Y":
                circuit.sdg(qubit)
                circuit.h(qubit)
            # "Z" and "I" need no basis change
    

        # --- convert user observable to SparsePauliOp ----------------------------
    @staticmethod
    def _to_sparse_pauli_op(observable) -> SparsePauliOp:
        """
        Accepts:
          - SparsePauliOp (returned as-is)
          - dict-like {label: coeff} where label is 'IXYZ...' string
        and returns a SparsePauliOp.
        """
        if isinstance(observable, SparsePauliOp):
            return observable

        # Simple fallback: mapping of labels to coeffs
        try:
            items = list(observable.items())
        except AttributeError as exc:
            raise TypeError(
                "Unsupported observable type; expected SparsePauliOp or dict-like."
            ) from exc

        if len(items) == 0:
            return SparsePauliOp(["I"], [0.0])

        labels, coeffs = zip(*items)
        return SparsePauliOp(list(labels), list(coeffs))

    # --- expectation for a single circuit and SparsePauliOp ------------------
    def _expectation_from_sparse_pauli_op(
        self, bound_circuit, observable: SparsePauliOp
    ) -> tuple[float, float]:
        """
        Compute ⟨O⟩ and an approximate std for a general SparsePauliOp O
        by sampling. Each Pauli term is measured separately
        (no grouping optimization here).
        """
        labels = observable.paulis.to_labels()
        coeffs = np.asarray(observable.coeffs, dtype=complex)

        exp_total = 0.0 + 0.0j
        var_total = 0.0

        for label, coeff in zip(labels, coeffs):
            if np.isclose(coeff, 0.0):
                continue

            # 1) Make a fresh circuit with the necessary basis changes
            term_circ = bound_circuit.copy()
            self._apply_basis_change_for_label(term_circ, label)

            # 2) Sample it
            counts = self._measure_and_get_counts(term_circ)
            shots = sum(counts.values())
            if shots == 0:
                continue

            # 3) Compute ⟨P⟩ for this Pauli string
            # Qiskit uses bitstrings where rightmost bit is qubit 0
            term_mean = 0.0
            for bitstring, c in counts.items():
                bits = bitstring[::-1]  # bits[i] corresponds to qubit i
                parity = 0
                for i, p in enumerate(label):
                    if p == "I":
                        continue
                    if i < len(bits) and bits[i] == "1":
                        parity ^= 1
                eig = -1 if parity else 1
                term_mean += eig * c

            term_mean /= shots

            # 4) Accumulate into total expectation and variance
            exp_total += coeff * term_mean

            # Eigenvalues of each Pauli string are ±1
            var_k = 1.0 - term_mean**2  # Var(P_k) ≈ 1 - ⟨P_k⟩²
            # Assume independent sampling for each term:
            var_total += (abs(coeff) ** 2) * var_k / shots

        exp_total = np.real_if_close(exp_total)
        std = float(np.sqrt(var_total)) if var_total > 0 else 0.0

        return float(exp_total), std


    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        """
        pub.circuit, pub.observables, pub.parameter_values, pub.precision
        are as in your original implementation.

        This version:
          * uses sampling via backend.get_counts()
          * supports general SparsePauliOp observables
          * avoids storing big broadcasted arrays of circuits/observables
        """
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision

        # Bind parameters once; result may be scalar or array-like of circuits
        bound_circuits = parameter_values.bind_all(circuit)

        # Lightweight broadcasting: doesn't create full arrays of objects
        bc = np.broadcast(bound_circuits, observables)

        # Result arrays: one scalar per (circuit, observable) pair
        evs = np.empty(bc.shape, dtype=np.float64)
        stds = np.empty(bc.shape, dtype=np.float64)

        for index, (bound_circuit, observable) in zip(np.ndindex(bc.shape), bc):
            # Ensure observable is a SparsePauliOp
            sp_op = self._to_sparse_pauli_op(observable)

            # Compute expectation and sampling std
            expectation_value, std = self._expectation_from_sparse_pauli_op(
                bound_circuit, sp_op
            )

            # Optional additional Gaussian noise (your original "precision" logic)
            if precision != 0:
                if not np.isreal(expectation_value):
                    raise ValueError(
                        "Given operator is not Hermitian and noise cannot be added."
                    )
                expectation_value = self._rng.normal(expectation_value, precision)

            evs[index] = expectation_value
            stds[index] = std

        data = DataBin(evs=evs, stds=stds, shape=evs.shape)
        return PubResult(
            data,
            metadata={
                "target_precision": precision,
                "circuit_metadata": pub.circuit.metadata,
            },
        )


class VLQ_Estimator(BaseEstimatorV2):

    def __init__(
        self,
        *,
        lexis_project: str = "vlq_demo_project",
        resource_name: str = "qaas_user",
        shots: int | None = 10000,
    ):

        self.session = LexisSession()
        self.lexis_project = lexis_project
        self.resource_name = resource_name
        self.token = self.session.get_access_token()
        self.provider = QProvider(self.token, self.lexis_project, self.resource_name)
        self.shots = shots
        self._shots = shots
        self._backend:QBackend = self.provider.get_backend()

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = 0
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]

        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs], metadata={"version": 2})

    def _measure_and_get_counts(self, circuit):
            """Measure all qubits in Z basis and return counts."""
            meas_circ = circuit.copy()
            meas_circ.measure_all()
            transpiled_qc = self._backend.transpile_to_IQM(meas_circ)
            counts = self.backend.run(transpiled_qc,
                                shots=self.shots).result().get_counts()

            return counts

    # --- basis change for a single Pauli label -------------------------------
    @staticmethod
    def _apply_basis_change_for_label(circuit, label: str):
        """
        Apply basis-change gates so that measuring in Z basis
        gives eigenvalues of the Pauli string `label`.

        Assumes `label[i]` acts on qubit i (Qiskit convention).
        X -> H
        Y -> Sdg + H
        Z/I -> nothing
        """
        for qubit, p in enumerate(label):
            if p == "X":
                circuit.h(qubit)
            elif p == "Y":
                circuit.sdg(qubit)
                circuit.h(qubit)
            # "Z" and "I" need no basis change
    

        # --- convert user observable to SparsePauliOp ----------------------------
    @staticmethod
    def _to_sparse_pauli_op(observable) -> SparsePauliOp:
        """
        Accepts:
          - SparsePauliOp (returned as-is)
          - dict-like {label: coeff} where label is 'IXYZ...' string
        and returns a SparsePauliOp.
        """
        if isinstance(observable, SparsePauliOp):
            return observable

        # Simple fallback: mapping of labels to coeffs
        try:
            items = list(observable.items())
        except AttributeError as exc:
            raise TypeError(
                "Unsupported observable type; expected SparsePauliOp or dict-like."
            ) from exc

        if len(items) == 0:
            return SparsePauliOp(["I"], [0.0])

        labels, coeffs = zip(*items)
        return SparsePauliOp(list(labels), list(coeffs))

    # --- expectation for a single circuit and SparsePauliOp ------------------
    def _expectation_from_sparse_pauli_op(
        self, bound_circuit, observable: SparsePauliOp
    ) -> tuple[float, float]:
        """
        Compute ⟨O⟩ and an approximate std for a general SparsePauliOp O
        by sampling. Each Pauli term is measured separately
        (no grouping optimization here).
        """
        labels = observable.paulis.to_labels()
        coeffs = np.asarray(observable.coeffs, dtype=complex)

        exp_total = 0.0 + 0.0j
        var_total = 0.0

        for label, coeff in zip(labels, coeffs):
            if np.isclose(coeff, 0.0):
                continue

            # 1) Make a fresh circuit with the necessary basis changes
            term_circ = bound_circuit.copy()
            self._apply_basis_change_for_label(term_circ, label)

            # 2) Sample it
            counts = self._measure_and_get_counts(term_circ)
            shots = sum(counts.values())
            if shots == 0:
                continue

            # 3) Compute ⟨P⟩ for this Pauli string
            # Qiskit uses bitstrings where rightmost bit is qubit 0
            term_mean = 0.0
            for bitstring, c in counts.items():
                bits = bitstring[::-1]  # bits[i] corresponds to qubit i
                parity = 0
                for i, p in enumerate(label):
                    if p == "I":
                        continue
                    if i < len(bits) and bits[i] == "1":
                        parity ^= 1
                eig = -1 if parity else 1
                term_mean += eig * c

            term_mean /= shots

            # 4) Accumulate into total expectation and variance
            exp_total += coeff * term_mean

            # Eigenvalues of each Pauli string are ±1
            var_k = 1.0 - term_mean**2  # Var(P_k) ≈ 1 - ⟨P_k⟩²
            # Assume independent sampling for each term:
            var_total += (abs(coeff) ** 2) * var_k / shots

        exp_total = np.real_if_close(exp_total)
        std = float(np.sqrt(var_total)) if var_total > 0 else 0.0

        return float(exp_total), std


    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        """
        pub.circuit, pub.observables, pub.parameter_values, pub.precision
        are as in your original implementation.

        This version:
          * uses sampling via backend.get_counts()
          * supports general SparsePauliOp observables
          * avoids storing big broadcasted arrays of circuits/observables
        """
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision

        # Bind parameters once; result may be scalar or array-like of circuits
        bound_circuits = parameter_values.bind_all(circuit)

        # Lightweight broadcasting: doesn't create full arrays of objects
        bc = np.broadcast(bound_circuits, observables)

        # Result arrays: one scalar per (circuit, observable) pair
        evs = np.empty(bc.shape, dtype=np.float64)
        stds = np.empty(bc.shape, dtype=np.float64)

        for index, (bound_circuit, observable) in zip(np.ndindex(bc.shape), bc):
            # Ensure observable is a SparsePauliOp
            sp_op = self._to_sparse_pauli_op(observable)

            # Compute expectation and sampling std
            expectation_value, std = self._expectation_from_sparse_pauli_op(
                bound_circuit, sp_op
            )

            # Optional additional Gaussian noise (your original "precision" logic)
            if precision != 0:
                if not np.isreal(expectation_value):
                    raise ValueError(
                        "Given operator is not Hermitian and noise cannot be added."
                    )
                expectation_value = self._rng.normal(expectation_value, precision)

            evs[index] = expectation_value
            stds[index] = std

        data = DataBin(evs=evs, stds=stds, shape=evs.shape)
        return PubResult(
            data,
            metadata={
                "target_precision": precision,
                "circuit_metadata": pub.circuit.metadata,
            },
        )



    

    
