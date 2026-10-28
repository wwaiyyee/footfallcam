"""Record installed versions in the dependency closure, not unrelated global packages."""
from importlib import metadata
from pathlib import Path
import platform

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    pending = ["staff-video[dev]", "setuptools", "wheel"]
    pinned = {}
    while pending:
        requested = Requirement(pending.pop())
        name = canonicalize_name(requested.name)
        if name in pinned:
            continue
        distribution = metadata.distribution(name)
        if name != "staff-video":
            pinned[name] = distribution.version
        extras = requested.extras | {""}
        for specification in distribution.requires or []:
            requirement = Requirement(specification)
            if requirement.marker is None or any(requirement.marker.evaluate({"extra": extra}) for extra in extras):
                pending.append(str(requirement))
    lines = [f"# Tested dependency closure: Python {platform.python_version()}, {platform.system()} {platform.machine()}.",
             "# Install on Python 3.9-3.12 with: python -m pip install -r requirements.lock",
             "# Torch platform-specific dependencies may differ; pyproject.toml remains the portable direct pin set."]
    lines += [f"{name}=={version}" for name, version in sorted(pinned.items())]
    Path("requirements.lock").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
