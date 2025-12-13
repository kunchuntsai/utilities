# C/C++ Utilities

A collection of portable tools for C/C++ projects. Each utility is self-contained, requires no external dependencies, and can be easily integrated into any C/C++ codebase.

## Design Philosophy

- **Portable**: Works on any platform with Python 3.6+
- **Zero Dependencies**: Uses only Python standard library
- **Self-Contained**: Each tool is independent and can be copied into any project
- **Easy Integration**: Simple CLI interface, suitable for CI/CD pipelines

## Available Utilities

| Utility | Description | Status |
|---------|-------------|--------|
| [analyzer](#dependency-analyzer) | C/C++ dependency analyzer with Clean Architecture analysis | Ready |

---

## Dependency Analyzer

Analyzes C/C++ header dependencies and generates interactive HTML visualizations with Clean Architecture layer analysis.

**Location**: `analyzer/`

### Features

- Dependency graph visualization (Doxygen-style with curved lines)
- Clean Architecture layer classification and violation detection
- Circular dependency detection
- Directory-level dependency analysis
- Interactive HTML report with D3.js

### Quick Start

```bash
# Install
cd analyzer
pip install .

# Run
cdep-analyzer /path/to/your/project
```

### Output

- `ca_layers.json` - Layer configuration (auto-generated, customizable)
- `dep_report.html` - Interactive dependency visualization

### Workflow

1. **First run**: Auto-detects dependencies and generates layer config
2. **Customize**: Edit `ca_layers.json` to match your architecture
3. **Run again**: Validates against Clean Architecture rules

See [analyzer/README.md](analyzer/README.md) for detailed documentation.

---

## Planned Utilities

The following tools are planned for future development:

| Utility | Description | Status |
|---------|-------------|--------|
| formatter | Code style formatter/checker | Planned |


---

## Installation

Each utility can be installed independently:

```bash
# Install a specific utility
cd <utility-name>
pip install .

# Or install in development mode
pip install -e .
```

Alternatively, copy the utility directory into your project and run directly:

```bash
python3 <utility-name>/<script>.py [options]
```

## Requirements

- Python 3.6 or higher
- No external packages required

## License

MIT License - See individual utility directories for license files.

## Contributing

Contributions are welcome! When adding a new utility, please ensure:

1. No external dependencies (stdlib only)
2. Python 3.6+ compatibility
3. Self-contained in its own directory
4. Includes README.md, LICENSE, and pyproject.toml
5. CLI interface with `--help` support
