import sys
from pathlib import Path

# Make `import marsad` work without installing the package.
sys.path.insert(0, str(Path(__file__).parent / "src"))
