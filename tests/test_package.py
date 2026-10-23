import importlib
import pkgutil

import eurolex


def test_all_modules_import() -> None:
    """Catches broken imports in the skeleton before any real code depends on them."""
    names = [m.name for m in pkgutil.walk_packages(eurolex.__path__, prefix="eurolex.")]
    assert "eurolex.retrieval.pipeline" in names
    for name in names:
        if name == "eurolex.ui.app":  # Streamlit scripts are run, not imported.
            continue
        importlib.import_module(name)
