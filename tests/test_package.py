from importlib.metadata import metadata

import quantcheck


def test_stable_package_version() -> None:
    assert quantcheck.__version__ == "1.0.0"


def test_runtime_metadata_declares_pyarrow_for_supported_columnar_inputs() -> None:
    package = metadata("quantcheck")
    assert package["Version"] == quantcheck.__version__
    requirements = package.get_all("Requires-Dist")
    assert requirements is not None
    assert any(requirement.startswith("pyarrow") for requirement in requirements)
