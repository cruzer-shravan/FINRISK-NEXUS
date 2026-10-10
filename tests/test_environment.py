import tomllib
from pathlib import Path


def test_project_configuration_is_valid():
    config_path = Path("pyproject.toml")

    assert config_path.exists()

    with config_path.open("rb") as file:
        config = tomllib.load(file)

    assert "tool" in config
    assert "ruff" in config["tool"]
    assert "pytest" in config["tool"]


def test_core_project_directories_exist():
    directories = [
        "data/raw",
        "data/bronze",
        "data/silver",
        "data/gold",
        "src/data",
        "src/features",
        "src/models",
        "src/pipelines",
        "api",
        "ui",
        "tests",
        "configs",
        "reports/figures",
    ]

    for directory in directories:
        assert Path(directory).is_dir(), f"Missing directory: {directory}"
