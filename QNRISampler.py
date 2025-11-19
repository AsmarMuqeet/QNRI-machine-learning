from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from qiskit.primitives.base import BaseSamplerV2
from qiskit.primitives.containers import (
    PrimitiveResult,
    SamplerPubLike,
    SamplerPubResult,
)
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qaas import QBackend, QProvider
from py4lexis.session import LexisSession

class fackec:
    def __init__(self,counts):
        self.counts = counts
    def get_counts(self):
        return self.counts

class fakedata:
    def __init__(self,counts):
        self.c = fackec(counts)

class customResult:
    def __init__(self,counts):
        self.data = fakedata(counts)

@dataclass
class Options:
    """Options for :class:`~.SamplerV2`."""

    backend_options: dict = field(default_factory=dict)
    """backend_options: Options passed to AerSimulator."""

    run_options: dict = field(default_factory=dict)
    """run_options: Options passed to run."""


class Aer_Sampler(BaseSamplerV2):
    """
    Each tuple of ``(circuit, <optional> parameter values, <optional> shots)``, called a sampler
    primitive unified bloc (PUB), produces its own array-valued result. The :meth:`~run` method can
    be given many pubs at once.

    * ``backend_options``: Options passed to AerSimulator.
      Default: {}.

    * ``run_options``: Options passed to :meth:`AerSimulator.run`.
      Default: {}.

    """

    def __init__(
        self,
        *,
        default_shots: int = 1024,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """
        Args:
            default_shots: The default shots for the sampler if not specified during run.
            seed: The seed for random number generation.
                If None, a random seeded default RNG will be used.
            options:
                the backend options (``backend_options``), and
                the runtime options (``run_options``).

        """
        self._default_shots = default_shots
        self._seed = seed

        self._options = Options(**options) if options else Options()
        self._backend = AerSimulator(**self.options.backend_options)
        self.pass_manager = generate_preset_pass_manager(backend=self._backend, optimization_level=3)

    @classmethod
    def from_backend(cls, backend, **options):
        """make new sampler that uses external backend"""
        sampler = cls(**options)
        if isinstance(backend, AerSimulator):
            sampler._backend = backend
        else:
            sampler._backend = AerSimulator.from_backend(backend)
        return sampler

    @property
    def default_shots(self) -> int:
        """Return the default shots"""
        return self._default_shots

    @property
    def seed(self) -> int | None:
        """Return the seed for random number generation."""
        return self._seed

    @property
    def options(self) -> Options:
        """Return the options"""
        return self._options

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        if shots is None:
            shots = self._default_shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _validate_pubs(self, pubs: list[SamplerPub]):
        for i, pub in enumerate(pubs):
            if len(pub.circuit.cregs) == 0:
                warnings.warn(
                    f"The {i}-th pub's circuit has no output classical registers and so the result "
                    "will be empty. Did you mean to add measurement instructions?",
                    UserWarning,
                )

    def _run(self, pubs: list[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        pub_dict = defaultdict(list)
        # consolidate pubs with the same number of shots
        for i, pub in enumerate(pubs):
            pub_dict[pub.shots].append(i)

        results = [None] * len(pubs)
        for shots, lst in pub_dict.items():
            # run pubs with the same number of shots at once
            pub_results = self._run_pubs([pubs[i] for i in lst], shots)
            # reconstruct the result of pubs
            for i, pub_result in zip(lst, pub_results):
                results[i] = pub_result
        return PrimitiveResult(results, metadata={"version": 2})


    def _run_pubs(self, pubs: list[SamplerPub], shots: int) -> list[SamplerPubResult]:
        """Compute results for pubs that all require the same value of ``shots``."""
        # Bind params onto each circuit up-front so the circuits are backend-agnostic.
        circuits = [
            _assign_parameters_to_circuit(pub.circuit, pub.parameter_values)
            for pub in pubs
        ]

        # adjust run_options not to overwrite existing options
        run_options = self.options.run_options.copy()
        for key in ["shots", "parameter_binds", "memory"]:
            if key in run_options:
                del run_options[key]
        if self._seed is not None and "seed_simulator" in run_options:
            del run_options["seed_simulator"]

        # run circuits (no parameter_binds)
        transpiled_circuits = self.pass_manager.run(circuits)
        result = self._backend.run(
            transpiled_circuits,
            shots=shots,
            seed_simulator=self._seed,
            **run_options,
        ).result()

        counts = result.get_counts()
        results = [customResult(count) for count in counts]
        return results

    

def _assign_parameters_to_circuit(circuit, parameter_values):
    """
    Return a new circuit with parameters assigned from `parameter_values`.

    Supports the common case of a *single* set of parameter values.
    If multiple sets are present (broadcasting/sweeps), it uses the first set.
    Expand sweeps upstream into multiple pubs if needed.
    """
    if not circuit.parameters:
        return circuit

    # Order matters: follow the circuit's parameter order.
    params = list(circuit.parameters)
    arr = parameter_values.as_array(params)

    # `arr` can be:
    #  - shape (len(params),)               -> single set
    #  - shape (1, len(params)) or (len(params), 1) -> single set with extra dim
    #  - shape (k, len(params))             -> k sets (sweep). We take the first row.
    if arr.ndim == 1:
        vector = arr
    else:
        # Take the first set if a sweep is present.
        vector = arr[0]

    # Build the mapping param -> scalar
    mapping = {p: float(vector[i]) for i, p in enumerate(params)}
    # Bind (non-inplace) so original circuit remains reusable on other backends.
    return circuit.assign_parameters(mapping, inplace=False)



def _convert_parameter_bindings(pub: SamplerPub) -> dict:
    circuit = pub.circuit
    parameter_values = pub.parameter_values
    parameter_binds = {}
    param_array = parameter_values.as_array(circuit.parameters)
    parameter_binds = {p: param_array[..., i].ravel() for i, p in enumerate(circuit.parameters)}
    return parameter_binds


class VLQ_Sampler(BaseSamplerV2):
    """
    Each tuple of ``(circuit, <optional> parameter values, <optional> shots)``, called a sampler
    primitive unified bloc (PUB), produces its own array-valued result. The :meth:`~run` method can
    be given many pubs at once.

    * ``backend_options``: Options passed to AerSimulator.
      Default: {}.

    * ``run_options``: Options passed to :meth:`AerSimulator.run`.
      Default: {}.

    """

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
        self._backend:QBackend = self.provider.get_backend()


    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        if shots is None:
            shots = self.shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _validate_pubs(self, pubs: list[SamplerPub]):
        for i, pub in enumerate(pubs):
            if len(pub.circuit.cregs) == 0:
                warnings.warn(
                    f"The {i}-th pub's circuit has no output classical registers and so the result "
                    "will be empty. Did you mean to add measurement instructions?",
                    UserWarning,
                )

    def _run(self, pubs: list[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        pub_dict = defaultdict(list)
        # consolidate pubs with the same number of shots
        for i, pub in enumerate(pubs):
            pub_dict[pub.shots].append(i)

        results = [None] * len(pubs)
        for shots, lst in pub_dict.items():
            # run pubs with the same number of shots at once
            pub_results = self._run_pubs([pubs[i] for i in lst], shots)
            # reconstruct the result of pubs
            for i, pub_result in zip(lst, pub_results):
                results[i] = pub_result
        return PrimitiveResult(results, metadata={"version": 2})


    def _run_pubs(self, pubs: list[SamplerPub], shots: int) -> list[SamplerPubResult]:
        """Compute results for pubs that all require the same value of ``shots``."""
        # Bind params onto each circuit up-front so the circuits are backend-agnostic.
        circuits = [
            _assign_parameters_to_circuit(pub.circuit, pub.parameter_values)
            for pub in pubs
        ]

        results = []

        for circuit in circuits:
            transpiled_qc = self._backend.transpile_to_IQM(circuit)
            counts = self.backend.run(transpiled_qc,
                                shots=self.shots).result().get_counts()
            results.append(customResult(counts))
        
        return results

