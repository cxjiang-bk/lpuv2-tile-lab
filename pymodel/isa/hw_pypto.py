"""Compile hw-native-sys PyPTO kernels and execute the native PTO C++ simulator."""

from dataclasses import dataclass, field
import hashlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
BACKEND = "hw_pto_cpp_sim"


def environment_details() -> dict:
    import pypto
    import pypto.language as pl

    expected = json.loads((ROOT / "data/upstream.lock.json").read_text())["pypto"]
    dist = metadata.distribution("pypto")
    if not hasattr(pl, "jit") or hasattr(pypto, "frontend") or dist.version != expected["version"]:
        raise RuntimeError("Use the locked hw-native-sys/pypto build; source scripts/hw-env.sh")
    runtime = metadata.distribution("simpler")
    from pypto.backend._ptoas_locate import find_ptoas_binary

    assembler = find_ptoas_binary()
    if assembler is None:
        raise RuntimeError("PTOAS is required for native execution; source scripts/hw-env.sh")
    version = subprocess.run([str(assembler), "--version"], capture_output=True, text=True, check=True)
    return {"backend": BACKEND, "pypto_version": dist.version,
            "pypto_source": expected["url"], "pypto_source_commit": expected["commit"],
            "installed_distribution": json.loads(dist.read_text("direct_url.json") or "{}"),
            "pypto_module": str(Path(pypto.__file__).resolve()),
            "runtime_version": runtime.version,
            "runtime_commit": expected["runtime"]["commit"],
            "runtime_installed_distribution": json.loads(runtime.read_text("direct_url.json") or "{}"),
            "ptoas": str(assembler), "ptoas_version": (version.stdout + version.stderr).strip(),
            "pto_isa_commit": expected["runtime"]["pto_isa_commit"],
            "execution": "HW PyPTO -> PTO IR -> PTOAS -> C++ shared libraries -> simpler a2a3sim",
            "execution_layer": "PTO C++ functional simulation", "platform": "a2a3sim",
            "hardware_precision_validated": False,
            "native_compilation": "deferred", "native_simulation": "deferred"}


@dataclass
class TensorProgram:
    kernel: object
    kernel_source: str
    details: dict
    native_outputs: list[Path] = field(default_factory=list)
    source_files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def compile(cls, kernel):
        details = environment_details()
        source = inspect.getsource(kernel._func)
        details["kernel_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        details["specialization"] = {}
        for name, value in inspect.getclosurevars(kernel._func).nonlocals.items():
            try:
                json.dumps(value)
                details["specialization"][name] = value
            except TypeError:
                details["specialization"][name] = str(value)
        sources = {str(p.relative_to(ROOT)): p.read_text() for p in (ROOT / "kernels/pypto").glob("*.py")}
        details["source_files_sha256"] = {name: hashlib.sha256(body.encode()).hexdigest()
                                          for name, body in sources.items()}
        return cls(kernel, source, details, source_files=sources)

    def execute(self, *tensors):
        if any(t.device.type != "cpu" or not t.is_contiguous() for t in tensors):
            raise ValueError("The native C++ simulator takes contiguous CPU tensors")
        from pypto.backend import BackendType
        from pypto.runtime import ExecutionMode, RunConfig

        config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.SIM,
                           dump_passes=True)
        started = time.perf_counter()
        compiled = self.kernel.compile(*tensors, config=config)
        directory = Path(compiled.output_dir)
        if not list(directory.rglob("*.pto")):
            raise RuntimeError("Compilation produced no PTO IR")
        # Execute the compiled artifact. Never substitute TorchCodegen or golden.
        compiled(*tensors, config=config)
        binaries = sorted(directory.rglob("*.so"))
        if not binaries:
            raise RuntimeError("Native simulation produced no compiled C++ shared libraries")
        self.native_outputs = [directory]
        self.details.update(native_compilation="completed", native_simulation="completed",
                            elapsed_seconds=time.perf_counter() - started,
                            generated_pto_count=len(list(directory.rglob("*.pto"))),
                            binary_sha256={str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in binaries})

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "kernel.py").write_text(self.kernel_source)
        for name, body in self.source_files.items():
            target = directory / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)
        for i, source in enumerate(self.native_outputs):
            shutil.copytree(source, directory / f"native-{i}", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
        (directory / "program.json").write_text(json.dumps(self.details, ensure_ascii=False, indent=2) + "\n")
