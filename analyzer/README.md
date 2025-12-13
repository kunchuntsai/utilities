# C/C++ Dependency Analyzer

Analyzes C/C++ header dependencies and Clean Architecture compliance.

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

## Installation

```bash
# Install from source
pip install .

# Or install in development mode
pip install -e .
```

## Usage

```bash
# After installation
cdep-analyzer /path/to/project

# Or run directly without installing
python3 cdep_analyzer.py /path/to/project
```

## Workflow

1. **First run**: Auto-detects dependencies and generates `ca_layers.json` with suggested layer assignments
2. **Edit config**: Modify `ca_layers.json` to match your design concept
3. **Run again**: Checks for Clean Architecture violations based on your configuration

## Output Files (in analyzer directory)

- `ca_layers.json` - Layer configuration
- `dep_report.html` - Interactive visualization

## Clean Architecture Layers

Order: outermost to innermost

| Layer | Description |
|-------|-------------|
| Presentation | Entry points, UI, drivers |
| Application | Use cases, orchestration |
| Core | Business logic, interfaces |
| Infrastructure | External services, utilities |

**Violation**: Inner layer depending on outer layer (e.g., Infrastructure -> Presentation)

## Configuration Format

```json
{
  "layers": [...],
  "directory_layers": {
    "Presentation": ["src"],
    "Application": ["src/core"],
    "Core": ["include"],
    "Infrastructure": ["src/utils", "examples"]
  },
  "file_overrides": {}
}
```
