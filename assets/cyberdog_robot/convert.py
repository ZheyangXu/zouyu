#!/usr/bin/env python3

"""Utility to convert the CyberDog xacro model into a URDF file."""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Optional, Sequence

import xacro

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _extend_ros_package_path(extra_paths: Iterable[Path]) -> None:
    """Ensure the provided paths are part of ROS_PACKAGE_PATH."""

    current = os.environ.get("ROS_PACKAGE_PATH", "")
    existing = [Path(p).resolve() for p in current.split(":") if p]
    for path in extra_paths:
        resolved = path.resolve()
        if resolved not in existing:
            existing.append(resolved)
    if existing:
        os.environ["ROS_PACKAGE_PATH"] = ":".join(str(p) for p in existing)


def _parse_mappings(raw_args: Iterable[str]) -> Dict[str, str]:
    mappings: Dict[str, str] = {}
    for item in raw_args:
        separator = ":=" if ":=" in item else "="
        try:
            key, value = item.split(separator, 1)
        except ValueError as exc:
            raise SystemExit(
                f"Failed to parse xacro argument '{item}'. Use name:=value format."
            ) from exc
        mappings[key] = value
    return mappings


def _default_output_path(xacro_path: Path) -> Path:
    return xacro_path.with_suffix(".urdf")


def _read_package_name(package_xml: Path) -> Optional[str]:
    try:
        tree = ET.parse(package_xml)
    except (ET.ParseError, OSError):
        return None
    name = tree.getroot().findtext("name")
    if not name:
        return None
    return name.strip()


def _discover_packages(search_roots: Iterable[Path]) -> Dict[str, Path]:
    packages: Dict[str, Path] = {}
    for root_path in search_roots:
        if not root_path.exists():
            continue
        try:
            package_files = root_path.rglob("package.xml")
        except OSError:
            continue
        for package_xml in package_files:
            package_root = package_xml.parent
            name = _read_package_name(package_xml)
            if not name:
                continue
            packages.setdefault(name, package_root)
    return packages


def _ensure_ament_index(search_roots: Iterable[Path]) -> None:
    try:
        import ament_index_python.packages  # type: ignore  # noqa: F401

        return
    except ImportError:
        pass

    combined_roots: list[Path] = []
    for root in search_roots:
        resolved = root.resolve()
        if resolved not in combined_roots:
            combined_roots.append(resolved)

    ros_package_path = os.environ.get("ROS_PACKAGE_PATH", "")
    for path in ros_package_path.split(":"):
        if not path:
            continue
        resolved = Path(path).resolve()
        if resolved not in combined_roots:
            combined_roots.append(resolved)

    packages = _discover_packages(combined_roots)

    class PackageNotFoundError(KeyError):
        """Mimic ament_index_python PackageNotFoundError."""

    def get_package_share_directory(pkg_name: str) -> str:
        try:
            return str(packages[pkg_name])
        except KeyError as exc:
            raise PackageNotFoundError(pkg_name) from exc

    def get_package_prefix(pkg_name: str) -> str:
        return get_package_share_directory(pkg_name)

    ament_pkg_module = ModuleType("ament_index_python.packages")
    ament_pkg_module.get_package_share_directory = get_package_share_directory  # type: ignore[attr-defined]
    ament_pkg_module.get_package_prefix = get_package_prefix  # type: ignore[attr-defined]
    ament_pkg_module.PackageNotFoundError = PackageNotFoundError  # type: ignore[attr-defined]

    ament_module = ModuleType("ament_index_python")
    ament_module.packages = ament_pkg_module  # type: ignore[attr-defined]
    ament_module.get_package_share_directory = get_package_share_directory  # type: ignore[attr-defined]
    ament_module.get_package_prefix = get_package_prefix  # type: ignore[attr-defined]

    sys.modules.setdefault("ament_index_python", ament_module)
    sys.modules.setdefault("ament_index_python.packages", ament_pkg_module)


def _prepare_ros_environment(search_roots: Iterable[Path]) -> None:
    paths = [root.resolve() for root in search_roots]
    _extend_ros_package_path(paths)
    _ensure_ament_index(paths)


def convert(xacro_path: Path, output_path: Path, mappings: Dict[str, str]) -> None:
    search_roots = {
        REPO_ROOT,
        SCRIPT_DIR,
        xacro_path.parent,
    }
    _prepare_ros_environment(search_roots)

    try:
        document: Any = xacro.process_file(str(xacro_path), mappings=mappings)
    except xacro.XacroException as exc:
        raise SystemExit(f"xacro processing failed: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(document, "toprettyxml"):
        xml = document.toprettyxml(indent="  ")
    elif hasattr(document, "toxml"):
        xml = document.toxml()
    else:
        xml = str(document)
    output_path.write_text(xml, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert robot.xacro into a URDF file."
    )
    parser.add_argument(
        "xacro_file",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent
        / "cyberdog_description"
        / "xacro"
        / "robot.xacro",
        help="Path to the input xacro file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Target URDF file. Defaults to the xacro filename with .urdf extension.",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        help="Additional xacro argument in the form name:=value. Repeat for multiple arguments.",
    )

    args = parser.parse_args(argv)
    xacro_path: Path = args.xacro_file.resolve()
    if not xacro_path.exists():
        raise SystemExit(f"Input xacro file not found: {xacro_path}")

    output_path = (
        args.output.resolve() if args.output else _default_output_path(xacro_path)
    )
    mappings = _parse_mappings(args.arg)

    convert(xacro_path, output_path, mappings)
    print(f"Wrote URDF to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
