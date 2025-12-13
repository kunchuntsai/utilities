#!/usr/bin/env python3
"""
C/C++ Dependency Analyzer with Clean Architecture Analysis

A portable tool for analyzing header dependencies in C/C++ projects.
Generates an interactive HTML visualization with Clean Architecture layer analysis.

Features:
  - Dependency graph visualization (Doxygen-style with curved lines)
  - Clean Architecture layer classification and violation detection
  - Circular dependency detection
  - Directory-level dependency analysis

Compatible with Python 3.6.3+

Usage:
    python3 cdep_analyzer.py /path/to/project

Output files (in analyzer directory):
    - ca_layers.json: Auto-generated layer config (edit to customize)
    - dep_report.html: Interactive dependency visualization

Author: Portable dependency analysis tool
"""

from __future__ import print_function

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

# Version check for Python 3.6+
if sys.version_info < (3, 6):
    print("Error: Python 3.6 or higher is required")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

# File extensions to scan
C_EXTENSIONS = {'.c', '.h'}
CPP_EXTENSIONS = {'.cpp', '.hpp', '.cc', '.hh', '.cxx', '.hxx', '.c++', '.h++'}
ALL_EXTENSIONS = C_EXTENSIONS | CPP_EXTENSIONS

# Common system headers to ignore (partial match)
SYSTEM_HEADER_PREFIXES = (
    # C standard library
    'stdio', 'stdlib', 'string', 'math', 'time', 'ctype', 'errno',
    'limits', 'stdbool', 'stdint', 'stddef', 'stdarg', 'signal',
    'assert', 'locale', 'setjmp', 'float', 'iso646', 'wchar', 'wctype',
    # POSIX
    'sys/', 'unistd', 'fcntl', 'pthread', 'dirent', 'termios',
    # C++ standard library
    'iostream', 'fstream', 'sstream', 'iomanip', 'vector', 'list',
    'map', 'set', 'unordered_map', 'unordered_set', 'array', 'deque',
    'queue', 'stack', 'algorithm', 'functional', 'iterator', 'memory',
    'utility', 'tuple', 'type_traits', 'chrono', 'ratio', 'thread',
    'mutex', 'condition_variable', 'future', 'atomic', 'regex',
    'random', 'numeric', 'complex', 'valarray', 'bitset', 'initializer_list',
    'typeinfo', 'typeindex', 'exception', 'stdexcept', 'new', 'limits',
    'climits', 'cfloat', 'cstdint', 'cstddef', 'cstdlib', 'cstring',
    'cmath', 'ctime', 'cctype', 'cerrno', 'cassert', 'cstdio', 'cstdarg',
    'csignal', 'clocale', 'csetjmp', 'cwchar', 'cwctype', 'cuchar',
    'cinttypes', 'cfenv', 'filesystem', 'optional', 'variant', 'any',
    'string_view', 'charconv', 'execution', 'span', 'ranges', 'format',
    'source_location', 'compare', 'version', 'numbers', 'bit', 'concepts',
    'coroutine', 'semaphore', 'latch', 'barrier', 'stop_token',
    # Platform specific
    'windows', 'Windows', 'win32', 'Win32', 'windef', 'winbase',
    'OpenCL/', 'CL/', 'cuda', 'CUDA', 'vulkan', 'Vulkan',
    'gl/', 'GL/', 'glm/', 'GLFW/', 'SDL', 'X11/', 'Cocoa/',
    # Common third-party
    'boost/', 'gtest/', 'gmock/', 'catch', 'doctest', 'json',
    'rapidjson/', 'nlohmann/', 'fmt/', 'spdlog/', 'eigen', 'Eigen/',
)

# Default directories to exclude
DEFAULT_EXCLUDES = {
    'build', 'cmake-build-debug', 'cmake-build-release',
    '.git', '.svn', '.hg',
    'node_modules', 'vendor', 'third_party', 'external', 'deps',
    '__pycache__', '.pytest_cache', '.tox',
}

# =============================================================================
# Clean Architecture Configuration (Dependency-Based)
# =============================================================================

# Layer definitions (outermost to innermost for intuitive reading)
# Standard Clean Architecture layers:
#   - Presentation: Entry points, UI, drivers (outermost)
#   - Application: Use cases, orchestration
#   - Core: Business logic, interfaces
#   - Infrastructure: External services, utilities (innermost)
CLEAN_ARCH_LAYERS = [
    {
        'name': 'Presentation',
        'description': 'Entry points, UI, drivers',
        'color': '#F44336',  # Red
    },
    {
        'name': 'Application',
        'description': 'Use cases, orchestration',
        'color': '#2196F3',  # Blue
    },
    {
        'name': 'Core',
        'description': 'Business logic, interfaces',
        'color': '#4CAF50',  # Green
    },
    {
        'name': 'Infrastructure',
        'description': 'External services, utilities',
        'color': '#FF9800',  # Orange
    },
]

# Default layer for edge cases
DEFAULT_LAYER = 'Application'


# =============================================================================
# Clean Architecture Analyzer (Dependency-Based Classification)
# =============================================================================

class CleanArchAnalyzer:
    """Analyzes code against Clean Architecture principles.

    Layer classification is based on actual dependency analysis:
    - Core (innermost): Files that are most depended upon, with few/no outgoing deps
    - Operations: Files with balanced fan-in/fan-out
    - Services: Utility files used across the codebase
    - HAL (outermost): Files that depend on many others but are rarely depended upon
    """

    def __init__(self, scanner, config_path=None):
        """
        Initialize the analyzer.

        Args:
            scanner: DependencyScanner instance with scanned data
            config_path: Optional path to custom layer config JSON file
        """
        self.scanner = scanner
        self.layers = CLEAN_ARCH_LAYERS.copy()
        self.file_layers = {}  # file_path -> layer_name
        self.file_depths = {}  # file_path -> dependency depth
        self.violations = []   # List of violation dicts (inner -> outer)
        self.warnings = []     # List of warning dicts (layer skipping)
        self.file_overrides = {}  # Manual overrides from config
        self.directory_layers = {}  # directory -> layer_name

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _load_config(self, config_path):
        """Load custom layer configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'layers' in config:
                self.layers = config['layers']

            if 'directory_layers' in config:
                # Format: {"LayerName": ["dir1", "dir2", ...]}
                for layer, dirs in config['directory_layers'].items():
                    if isinstance(dirs, list):
                        for d in dirs:
                            self.directory_layers[d] = layer
                    else:
                        # Support single string for backwards compatibility
                        self.directory_layers[dirs] = layer

            if 'file_overrides' in config:
                for path, layer in config['file_overrides'].items():
                    if not path.startswith('#'):  # Skip comment entries
                        self.file_overrides[path] = layer

        except (IOError, json.JSONDecodeError) as e:
            print("Warning: Could not load config {}: {}".format(config_path, e))

    def _get_layer_for_file(self, file_path):
        """Get layer for a file based on config (file override > directory > None)."""
        # 1. Check file-specific override first
        if file_path in self.file_overrides:
            return self.file_overrides[file_path]

        # 2. Check directory-based assignment (most specific directory wins)
        file_dir = os.path.dirname(file_path) or '.'
        best_match = None
        best_match_len = -1

        for dir_pattern, layer in self.directory_layers.items():
            # Check if file is in this directory or subdirectory
            if file_dir == dir_pattern or file_dir.startswith(dir_pattern + '/'):
                # Use longest match (most specific directory)
                if len(dir_pattern) > best_match_len:
                    best_match = layer
                    best_match_len = len(dir_pattern)

        return best_match  # None if no match

    def _get_layer_index(self, layer_name):
        """Get the index of a layer (lower = outer, higher = inner)."""
        for idx, layer in enumerate(self.layers):
            if layer['name'] == layer_name:
                return idx
        return -1

    def _calculate_dependency_depth(self):
        """Calculate dependency depth for each file using topological analysis.

        Depth 0 = files with no outgoing dependencies (leaf nodes, most fundamental)
        Higher depth = files that depend on others
        """
        depths = {}
        visited = set()
        in_progress = set()

        def calc_depth(file_path):
            if file_path in depths:
                return depths[file_path]
            if file_path in in_progress:
                # Cycle detected, return 0 to break it
                return 0
            if file_path not in self.scanner.files:
                return 0

            in_progress.add(file_path)
            deps = self.scanner.dependencies.get(file_path, set())

            if not deps:
                depth = 0
            else:
                max_dep_depth = 0
                for dep in deps:
                    if dep in self.scanner.files:
                        max_dep_depth = max(max_dep_depth, calc_depth(dep) + 1)
                depth = max_dep_depth

            in_progress.discard(file_path)
            depths[file_path] = depth
            return depth

        for file_path in self.scanner.files:
            calc_depth(file_path)

        return depths

    def _classify_by_dependency(self):
        """Classify files into layers based on dependency analysis."""
        # Calculate dependency depths
        self.file_depths = self._calculate_dependency_depth()

        if not self.file_depths:
            return

        # Get depth statistics
        all_depths = list(self.file_depths.values())
        max_depth = max(all_depths) if all_depths else 0

        # Define layer boundaries based on depth distribution
        # Layer order (outermost to innermost): Presentation, Application, Core, Infrastructure
        # Presentation: highest depth (entry points, depend on many)
        # Application: medium-high depth (use cases)
        # Core: low depth, high fan-in (interfaces, business logic)
        # Infrastructure: depth 0, high fan-in (utilities, no outgoing deps)

        num_layers = len(self.layers)

        for file_path in self.scanner.files:
            # Check for config-based assignment first (file override or directory)
            config_layer = self._get_layer_for_file(file_path)
            if config_layer:
                self.file_layers[file_path] = config_layer
                continue

            depth = self.file_depths.get(file_path, 0)
            fan_in = len(self.scanner.reverse_deps.get(file_path, set()))
            fan_out = len(self.scanner.dependencies.get(file_path, set()))

            # Determine layer based on dependency characteristics
            # Higher layer_idx = inner layer (Infrastructure)
            # Lower layer_idx = outer layer (Presentation)
            if max_depth == 0:
                # All files have no dependencies, put in Infrastructure (innermost)
                layer_idx = num_layers - 1
            elif depth == 0:
                # No outgoing dependencies = Infrastructure (innermost, foundational)
                layer_idx = num_layers - 1
            elif fan_in > fan_out and fan_in > 2:
                # High fan-in (many dependents) = Core (interfaces/utilities)
                layer_idx = num_layers - 2
            elif depth >= max_depth - 1 and fan_out > fan_in:
                # Highest depth, more deps than dependents = Presentation (entry points)
                layer_idx = 0
            else:
                # Scale remaining to Application/Core range
                if max_depth > 1:
                    # Invert: higher depth = lower index (outer)
                    normalized = 1.0 - (depth / max_depth)
                    layer_idx = int(normalized * (num_layers - 2)) + 1
                    layer_idx = max(1, min(layer_idx, num_layers - 2))
                else:
                    layer_idx = 1  # Application

            self.file_layers[file_path] = self.layers[layer_idx]['name']

    def analyze(self):
        """Analyze dependencies and classify files into Clean Architecture layers."""
        if not self.scanner:
            return self

        # Classify files based on dependency analysis
        self._classify_by_dependency()

        # Find violations (inner layer depending on outer layer)
        # With dependency-based classification, violations should be minimal
        self.violations = []
        self.warnings = []
        for src_file, deps in self.scanner.dependencies.items():
            src_layer = self.file_layers.get(src_file, self.layers[0]['name'])
            src_idx = self._get_layer_index(src_layer)

            for dep_file in deps:
                dep_layer = self.file_layers.get(dep_file, self.layers[0]['name'])
                dep_idx = self._get_layer_index(dep_layer)

                # Violation: inner layer (higher index) depends on outer layer (lower index)
                if src_idx > dep_idx:
                    self.violations.append({
                        'source': src_file,
                        'target': dep_file,
                        'source_layer': src_layer,
                        'target_layer': dep_layer,
                        'severity': src_idx - dep_idx,
                    })
                # Warning: layer skipping (outer depends on non-adjacent inner)
                elif dep_idx - src_idx > 1:
                    self.warnings.append({
                        'source': src_file,
                        'target': dep_file,
                        'source_layer': src_layer,
                        'target_layer': dep_layer,
                        'skipped_layers': dep_idx - src_idx - 1,
                    })

        return self

    def get_layer_stats(self):
        """Get statistics by layer."""
        stats = {}
        for layer in self.layers:
            layer_name = layer['name']
            files = [f for f, l in self.file_layers.items() if l == layer_name]
            stats[layer_name] = {
                'files': len(files),
                'lines': sum(self.scanner.files[f]['line_count'] for f in files if f in self.scanner.files),
                'color': layer['color'],
                'description': layer['description'],
            }
        return stats

    def get_violation_summary(self):
        """Get summary of violations by layer pair."""
        summary = defaultdict(int)
        for v in self.violations:
            key = "{} -> {}".format(v['source_layer'], v['target_layer'])
            summary[key] += 1
        return dict(summary)

    def get_warning_summary(self):
        """Get summary of warnings (layer skipping) by layer pair."""
        summary = defaultdict(int)
        for w in self.warnings:
            key = "{} -> {}".format(w['source_layer'], w['target_layer'])
            summary[key] += 1
        return dict(summary)

    def generate_config_template(self, output_path):
        """Generate a configuration file with auto-detected layer assignments by directory."""
        # Group files by directory and find dominant layer for each
        dir_layers = {}  # directory -> {layer: count}
        for f, layer in self.file_layers.items():
            d = os.path.dirname(f) or '.'
            if d not in dir_layers:
                dir_layers[d] = {}
            dir_layers[d][layer] = dir_layers[d].get(layer, 0) + 1

        # Determine dominant layer for each directory
        dir_dominant = {}
        for d, layer_counts in dir_layers.items():
            dominant = max(layer_counts.items(), key=lambda x: x[1])[0]
            dir_dominant[d] = dominant

        # Group directories by layer: {"LayerName": ["dir1", "dir2"]}
        # Build in layer order (Presentation, Application, Core, Infrastructure)
        # Keep all layers even if empty, so users can see the full structure
        layer_dirs = {}
        for layer_def in self.layers:
            layer_name = layer_def['name']
            layer_dirs[layer_name] = []
        for d, layer in sorted(dir_dominant.items()):
            layer_dirs[layer].append(d)

        config = {
            '_comment': 'Layer order: outermost to innermost. Edit directory_layers to match your design.',
            'layers': [
                {
                    'name': layer['name'],
                    'description': layer['description'],
                    'color': layer['color'],
                }
                for layer in self.layers
            ],
            'directory_layers': layer_dirs,
            'file_overrides': {},
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        return output_path


# =============================================================================
# Dependency Scanner
# =============================================================================

class DependencyScanner:
    """Scans C/C++ files for dependencies."""

    def __init__(self, root_path, exclude_dirs=None, include_system=False):
        """
        Initialize the scanner.

        Args:
            root_path: Path to the project root directory
            exclude_dirs: Set of directory names to exclude
            include_system: Whether to include system headers
        """
        self.root_path = os.path.abspath(root_path)
        self.exclude_dirs = exclude_dirs or DEFAULT_EXCLUDES
        self.include_system = include_system

        # Regex for #include statements
        self.include_pattern = re.compile(
            r'^\s*#\s*include\s+([<"])([^>"]+)[>"]',
            re.MULTILINE
        )

        # Storage
        self.files = {}  # file_path -> FileInfo
        self.dependencies = defaultdict(set)  # file -> set of included files
        self.reverse_deps = defaultdict(set)  # file -> set of files that include it
        self.unresolved = defaultdict(set)  # file -> set of unresolved includes

    def scan(self):
        """Scan the project and build dependency graph."""
        # Find all source files
        self._find_files()

        # Parse includes
        self._parse_includes()

        # Resolve dependencies
        self._resolve_dependencies()

        return self

    def _find_files(self):
        """Find all C/C++ files in the project."""
        for dirpath, dirnames, filenames in os.walk(self.root_path):
            # Filter out excluded directories
            dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in ALL_EXTENSIONS:
                    full_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(full_path, self.root_path)

                    self.files[rel_path] = {
                        'full_path': full_path,
                        'rel_path': rel_path,
                        'filename': filename,
                        'extension': ext,
                        'directory': os.path.relpath(dirpath, self.root_path),
                        'is_header': ext in {'.h', '.hpp', '.hh', '.hxx', '.h++'},
                        'raw_includes': [],
                        'line_count': 0,
                    }

    def _parse_includes(self):
        """Parse #include statements from all files."""
        for rel_path, info in self.files.items():
            try:
                with open(info['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Count lines
                info['line_count'] = content.count('\n') + 1

                # Find includes
                for match in self.include_pattern.finditer(content):
                    bracket_type = match.group(1)  # < or "
                    include_path = match.group(2)

                    is_system = (bracket_type == '<')

                    # Skip system headers if not requested
                    if is_system and not self.include_system:
                        if include_path.startswith(SYSTEM_HEADER_PREFIXES):
                            continue

                    info['raw_includes'].append({
                        'path': include_path,
                        'is_system': is_system,
                    })
            except (IOError, OSError) as e:
                print("Warning: Could not read {}: {}".format(rel_path, e))

    def _resolve_dependencies(self):
        """Resolve include paths to actual files."""
        # Build lookup maps
        filename_map = defaultdict(list)  # filename -> list of rel_paths
        for rel_path in self.files:
            filename = os.path.basename(rel_path)
            filename_map[filename].append(rel_path)

        for rel_path, info in self.files.items():
            file_dir = os.path.dirname(rel_path)

            for inc in info['raw_includes']:
                inc_path = inc['path']
                resolved = None

                # Try to resolve the include
                # 1. Relative to current file
                candidate = os.path.normpath(os.path.join(file_dir, inc_path))
                if candidate in self.files:
                    resolved = candidate

                # 2. From project root
                if resolved is None:
                    if inc_path in self.files:
                        resolved = inc_path

                # 3. By filename match
                if resolved is None:
                    basename = os.path.basename(inc_path)
                    candidates = filename_map.get(basename, [])

                    if len(candidates) == 1:
                        resolved = candidates[0]
                    elif len(candidates) > 1:
                        # Try to find best match by path similarity
                        for c in candidates:
                            if c.endswith(inc_path):
                                resolved = c
                                break
                        if resolved is None:
                            # Use first match
                            resolved = candidates[0]

                if resolved:
                    self.dependencies[rel_path].add(resolved)
                    self.reverse_deps[resolved].add(rel_path)
                elif not inc['is_system']:
                    self.unresolved[rel_path].add(inc_path)

    def get_stats(self):
        """Get statistics about the scan."""
        total_files = len(self.files)
        headers = sum(1 for f in self.files.values() if f['is_header'])
        sources = total_files - headers
        total_lines = sum(f['line_count'] for f in self.files.values())
        total_deps = sum(len(d) for d in self.dependencies.values())

        return {
            'total_files': total_files,
            'header_files': headers,
            'source_files': sources,
            'total_lines': total_lines,
            'total_dependencies': total_deps,
            'files_with_deps': len(self.dependencies),
            'unresolved_includes': sum(len(u) for u in self.unresolved.values()),
        }

    def find_cycles(self):
        """Find circular dependencies."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.dependencies.get(node, set()):
                if neighbor not in visited:
                    cycle = dfs(neighbor)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            path.pop()
            rec_stack.remove(node)
            return None

        for node in self.files:
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    # Normalize cycle (start from smallest element)
                    min_idx = cycle.index(min(cycle[:-1]))
                    normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                    if normalized not in cycles:
                        cycles.append(normalized)

        return cycles

    def get_directory_deps(self):
        """Get dependencies aggregated by directory."""
        dir_deps = defaultdict(lambda: defaultdict(int))

        for src_file, deps in self.dependencies.items():
            src_dir = os.path.dirname(src_file) or '.'
            for dep_file in deps:
                dep_dir = os.path.dirname(dep_file) or '.'
                if src_dir != dep_dir:
                    dir_deps[src_dir][dep_dir] += 1

        return dict(dir_deps)

    def get_most_included(self, limit=20):
        """Get most frequently included files."""
        counts = [(f, len(deps)) for f, deps in self.reverse_deps.items()]
        counts.sort(key=lambda x: -x[1])
        return counts[:limit]

    def get_most_including(self, limit=20):
        """Get files with most includes."""
        counts = [(f, len(deps)) for f, deps in self.dependencies.items()]
        counts.sort(key=lambda x: -x[1])
        return counts[:limit]


# =============================================================================
# HTML Report Generator
# =============================================================================

def generate_html_report(scanner, output_path, clean_arch_analyzer=None):
    """Generate an interactive HTML report."""

    stats = scanner.get_stats()
    cycles = scanner.find_cycles()
    dir_deps = scanner.get_directory_deps()
    most_included = scanner.get_most_included(15)
    most_including = scanner.get_most_including(15)

    # Clean Architecture data
    if clean_arch_analyzer:
        ca_layer_stats = clean_arch_analyzer.get_layer_stats()
        ca_violations = clean_arch_analyzer.violations
        ca_warnings = clean_arch_analyzer.warnings
        ca_layers = clean_arch_analyzer.layers
        ca_file_layers = clean_arch_analyzer.file_layers
    else:
        ca_layer_stats = {}
        ca_violations = []
        ca_warnings = []
        ca_layers = []
        ca_file_layers = {}

    # Prepare graph data for D3.js
    nodes = []
    node_index = {}

    # Get all directories
    directories = set()
    for rel_path in scanner.files:
        dir_name = os.path.dirname(rel_path) or '.'
        directories.add(dir_name)

    # Create nodes for files
    for idx, (rel_path, info) in enumerate(scanner.files.items()):
        node_index[rel_path] = idx
        dir_name = os.path.dirname(rel_path) or '.'
        layer = ca_file_layers.get(rel_path, DEFAULT_LAYER)
        layer_color = next((l['color'] for l in ca_layers if l['name'] == layer), '#888888')
        nodes.append({
            'id': idx,
            'name': info['filename'],
            'path': rel_path,
            'directory': dir_name,
            'isHeader': info['is_header'],
            'lines': info['line_count'],
            'fanIn': len(scanner.reverse_deps.get(rel_path, set())),
            'fanOut': len(scanner.dependencies.get(rel_path, set())),
            'layer': layer,
            'layerColor': layer_color,
        })

    # Create links with violation info
    links = []
    violation_set = {(v['source'], v['target']) for v in ca_violations}
    for src_file, deps in scanner.dependencies.items():
        src_idx = node_index.get(src_file)
        if src_idx is not None:
            for dep_file in deps:
                tgt_idx = node_index.get(dep_file)
                if tgt_idx is not None:
                    is_violation = (src_file, dep_file) in violation_set
                    links.append({
                        'source': src_idx,
                        'target': tgt_idx,
                        'isViolation': is_violation,
                    })

    # Directory summary
    dir_summary = []
    for dir_name in sorted(directories):
        files_in_dir = [f for f, i in scanner.files.items()
                       if (os.path.dirname(f) or '.') == dir_name]
        lines = sum(scanner.files[f]['line_count'] for f in files_in_dir)
        dir_summary.append({
            'name': dir_name,
            'files': len(files_in_dir),
            'lines': lines,
        })

    # Generate HTML
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>C/C++ Dependency Analysis Report</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e;
            color: #eee;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 30px;
        }}
        header h1 {{
            color: #00d4ff;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        header .subtitle {{
            color: #888;
            font-size: 1.1em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #333;
        }}
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #00d4ff;
        }}
        .stat-card .label {{
            color: #888;
            margin-top: 5px;
        }}
        .section {{
            background: #16213e;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #333;
        }}
        .section h2 {{
            color: #00d4ff;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        .graph-container {{
            width: 100%;
            height: 600px;
            background: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }}
        .graph-controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 10;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .graph-controls button, .graph-controls select {{
            background: #f0f0f0;
            color: #333;
            border: 1px solid #999;
            padding: 8px 15px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 13px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .graph-controls button:hover {{
            background: #e0e0e0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            color: #00d4ff;
            font-weight: 600;
        }}
        tr:hover {{
            background: rgba(0, 212, 255, 0.1);
        }}
        .bar {{
            background: #00d4ff;
            height: 20px;
            border-radius: 3px;
            min-width: 5px;
        }}
        .bar-container {{
            background: #333;
            border-radius: 3px;
            overflow: hidden;
            width: 200px;
        }}
        .cycle-warning {{
            background: #5a1a1a;
            border: 1px solid #ff4444;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .cycle-warning h3 {{
            color: #ff4444;
            margin-bottom: 10px;
        }}
        .cycle-path {{
            font-family: monospace;
            background: #2a1a1a;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            overflow-x: auto;
        }}
        .no-cycles {{
            color: #44ff44;
            padding: 15px;
            background: #1a3a1a;
            border-radius: 8px;
            border: 1px solid #44ff44;
        }}
        .tooltip {{
            position: absolute;
            background: #16213e;
            border: 1px solid #00d4ff;
            border-radius: 5px;
            padding: 10px;
            pointer-events: none;
            font-size: 12px;
            max-width: 300px;
            z-index: 100;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
        .tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
        }}
        .tab {{
            background: #0f0f23;
            border: 1px solid #333;
            color: #888;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
        }}
        .tab.active {{
            background: #16213e;
            color: #00d4ff;
            border-bottom-color: #16213e;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #333;
            margin-top: 30px;
        }}
        /* Clean Architecture Styles */
        .ca-section {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }}
        .ca-rings {{
            position: relative;
            width: 100%;
            height: 600px;
            background: #ffffff;
            border-radius: 8px;
            overflow: hidden;
        }}
        .layer-legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
            padding: 15px;
            background: #f8f8f8;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .layer-legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: #333;
        }}
        .layer-legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }}
        .violation-table {{
            width: 100%;
            margin-top: 20px;
        }}
        .violation-table th {{
            background: #1a1a2e;
        }}
        .violation-row {{
            background: rgba(244, 67, 54, 0.1);
        }}
        .violation-row:hover {{
            background: rgba(244, 67, 54, 0.2);
        }}
        .violation-badge {{
            background: #F44336;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
        }}
        .ok-badge {{
            background: #4CAF50;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
        }}
        .layer-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .layer-stat-card {{
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 2px solid;
        }}
        .layer-stat-card .layer-name {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 5px;
        }}
        .layer-stat-card .layer-desc {{
            font-size: 0.85em;
            opacity: 0.8;
            margin-bottom: 10px;
        }}
        .layer-stat-card .layer-files {{
            font-size: 1.5em;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Dependency Analysis Report</h1>
            <div class="subtitle">
                Project: {project_path}<br>
                Generated: {timestamp}
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{total_files}</div>
                <div class="label">Total Files</div>
            </div>
            <div class="stat-card">
                <div class="value">{header_files}</div>
                <div class="label">Header Files</div>
            </div>
            <div class="stat-card">
                <div class="value">{source_files}</div>
                <div class="label">Source Files</div>
            </div>
            <div class="stat-card">
                <div class="value">{total_lines:,}</div>
                <div class="label">Lines of Code</div>
            </div>
            <div class="stat-card">
                <div class="value">{total_deps}</div>
                <div class="label">Dependencies</div>
            </div>
            <div class="stat-card">
                <div class="value">{cycle_count}</div>
                <div class="label">Circular Deps</div>
            </div>
        </div>

        <div class="section">
            <h2>Dependency Graph</h2>
            <div class="graph-container" id="graph">
                <div class="graph-controls">
                    <select id="dirFilter">
                        <option value="">All Directories</option>
                        {dir_options}
                    </select>
                    <button onclick="resetZoom()">Reset View</button>
                    <button onclick="toggleLabels()">Toggle Labels</button>
                </div>
            </div>
            <div class="legend" style="background: #f8f8f8; padding: 10px; border-radius: 4px; border: 1px solid #ddd;">
                <div class="legend-item">
                    <div class="legend-color" style="background: linear-gradient(180deg, #bfdfff 0%, #a8c8e8 100%); border: 1px solid #84b0c7; border-radius: 2px;"></div>
                    <span style="color: #333;">Header File (.h/.hpp)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: linear-gradient(180deg, #ffffd0 0%, #f0e68c 100%); border: 1px solid #c0a000; border-radius: 2px;"></div>
                    <span style="color: #333;">Source File (.c/.cpp)</span>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Circular Dependencies</h2>
            {cycles_html}
        </div>

        <div class="section ca-section">
            <h2>Clean Architecture Analysis</h2>
            <p style="color: #666; margin-bottom: 15px;">
                Module-level dependencies organized by Clean Architecture layers.
                Dependencies should flow: Presentation &rarr; Application &rarr; Core &larr; Infrastructure
                <span class="violation-badge" style="margin-left: 10px;">{violation_count} violations</span>
            </p>

            <div class="layer-stats">
                {layer_stats_html}
            </div>

            <div class="graph-container" id="ca-graph" style="height: 500px;"></div>

            <div class="layer-legend">
                {layer_legend_html}
            </div>

            {violations_html}
        </div>

        <div class="section">
            <h2>File Analysis</h2>
            <div class="tabs">
                <div class="tab active" onclick="showTab('most-included')">Most Included</div>
                <div class="tab" onclick="showTab('most-including')">Most Includes</div>
                <div class="tab" onclick="showTab('directories')">Directories</div>
            </div>

            <div id="most-included" class="tab-content active">
                <table>
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>Included By</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody>
                        {most_included_rows}
                    </tbody>
                </table>
            </div>

            <div id="most-including" class="tab-content">
                <table>
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>Includes</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody>
                        {most_including_rows}
                    </tbody>
                </table>
            </div>

            <div id="directories" class="tab-content">
                <table>
                    <thead>
                        <tr>
                            <th>Directory</th>
                            <th>Files</th>
                            <th>Lines</th>
                        </tr>
                    </thead>
                    <tbody>
                        {dir_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <footer>
            Generated by C/C++ Dependency Analyzer | Python {python_version}
        </footer>
    </div>

    <div id="tooltip" class="tooltip" style="display: none;"></div>

    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        // Data
        const nodes = {nodes_json};
        const links = {links_json};
        const dirDeps = {dir_deps_json};
        const caLayers = {ca_layers_json};
        const caViolations = {ca_violations_json};

        // Settings
        let showLabels = true;
        let currentFilter = '';

        // File Graph - Doxygen Style Layered Layout
        const graphContainer = document.getElementById('graph');
        const width = graphContainer.clientWidth;
        const height = graphContainer.clientHeight;

        const svg = d3.select('#graph')
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Doxygen-style gradients
        const defs = svg.append('defs');

        // Header file gradient (blue)
        const headerGradient = defs.append('linearGradient')
            .attr('id', 'headerGradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '0%').attr('y2', '100%');
        headerGradient.append('stop').attr('offset', '0%').attr('stop-color', '#bfdfff');
        headerGradient.append('stop').attr('offset', '100%').attr('stop-color', '#a8c8e8');

        // Source file gradient (yellow)
        const sourceGradient = defs.append('linearGradient')
            .attr('id', 'sourceGradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '0%').attr('y2', '100%');
        sourceGradient.append('stop').attr('offset', '0%').attr('stop-color', '#ffffd0');
        sourceGradient.append('stop').attr('offset', '100%').attr('stop-color', '#f0e68c');

        // Highlight gradient (green)
        const highlightGradient = defs.append('linearGradient')
            .attr('id', 'highlightGradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '0%').attr('y2', '100%');
        highlightGradient.append('stop').attr('offset', '0%').attr('stop-color', '#d0ffd0');
        highlightGradient.append('stop').attr('offset', '100%').attr('stop-color', '#90ee90');

        // Arrow marker
        defs.append('marker')
            .attr('id', 'doxygen-arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 10)
            .attr('refY', 0)
            .attr('markerWidth', 8)
            .attr('markerHeight', 8)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-4L10,0L0,4')
            .attr('fill', '#606060');

        const g = svg.append('g');

        // Zoom
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => g.attr('transform', event.transform));

        svg.call(zoom);

        function resetZoom() {{
            svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
        }}

        function toggleLabels() {{
            showLabels = !showLabels;
            g.selectAll('.node-label').style('display', showLabels ? 'block' : 'none');
        }}

        // Build adjacency lists for layer calculation
        const nodeById = new Map(nodes.map(n => [n.id, n]));
        const outgoing = new Map();
        const incoming = new Map();

        nodes.forEach(n => {{
            outgoing.set(n.id, new Set());
            incoming.set(n.id, new Set());
        }});

        links.forEach(l => {{
            const srcId = typeof l.source === 'object' ? l.source.id : l.source;
            const tgtId = typeof l.target === 'object' ? l.target.id : l.target;
            outgoing.get(srcId).add(tgtId);
            incoming.get(tgtId).add(srcId);
        }});

        // Calculate layers using longest path algorithm
        const layers = new Map();
        const visited = new Set();

        function calculateLayer(nodeId) {{
            if (layers.has(nodeId)) return layers.get(nodeId);
            if (visited.has(nodeId)) return 0;

            visited.add(nodeId);
            const deps = outgoing.get(nodeId);

            if (!deps || deps.size === 0) {{
                layers.set(nodeId, 0);
                return 0;
            }}

            let maxDepLayer = 0;
            deps.forEach(depId => {{
                maxDepLayer = Math.max(maxDepLayer, calculateLayer(depId) + 1);
            }});

            layers.set(nodeId, maxDepLayer);
            return maxDepLayer;
        }}

        nodes.forEach(n => calculateLayer(n.id));

        // Group nodes by layer
        const layerGroups = new Map();
        let maxLayer = 0;

        nodes.forEach(n => {{
            const layer = layers.get(n.id) || 0;
            maxLayer = Math.max(maxLayer, layer);
            if (!layerGroups.has(layer)) layerGroups.set(layer, []);
            layerGroups.get(layer).push(n);
        }});

        // Doxygen-style node dimensions
        const nodeHeight = 26;
        const nodeMinWidth = 100;
        const nodePadding = 10;
        const layerSpacing = 70;
        const nodeSpacingH = 20;

        // Calculate node widths based on text
        const tempText = svg.append('text')
            .attr('font-family', 'Consolas, Monaco, monospace')
            .attr('font-size', '11px');

        nodes.forEach(n => {{
            tempText.text(n.name);
            n.textWidth = tempText.node().getComputedTextLength();
            n.nodeWidth = Math.max(nodeMinWidth, n.textWidth + nodePadding * 2);
        }});

        tempText.remove();

        // Initial sort by directory/name
        layerGroups.forEach((nodesInLayer, layer) => {{
            nodesInLayer.sort((a, b) => {{
                if (a.directory !== b.directory) return a.directory.localeCompare(b.directory);
                return a.name.localeCompare(b.name);
            }});
        }});

        // Position nodes helper
        const padding = 60;
        const startY = padding;

        function positionLayer(layer) {{
            const nodesInLayer = layerGroups.get(layer);
            if (!nodesInLayer) return;
            const y = startY + (maxLayer - layer) * layerSpacing;
            const totalWidth = nodesInLayer.reduce((sum, n) => sum + n.nodeWidth, 0) + (nodesInLayer.length - 1) * nodeSpacingH;
            let x = (width - totalWidth) / 2;
            nodesInLayer.forEach(n => {{
                n.x = x + n.nodeWidth / 2;
                n.y = y;
                n.graphDepth = layer;
                x += n.nodeWidth + nodeSpacingH;
            }});
        }}

        // Initial positioning
        for (let layer = 0; layer <= maxLayer; layer++) {{
            positionLayer(layer);
        }}

        // Barycenter crossing reduction (multiple passes)
        for (let iter = 0; iter < 4; iter++) {{
            // Down pass: reorder based on neighbors above (higher layer = dependencies)
            for (let layer = maxLayer - 1; layer >= 0; layer--) {{
                const nodesInLayer = layerGroups.get(layer);
                if (!nodesInLayer || nodesInLayer.length < 2) continue;

                nodesInLayer.forEach(n => {{
                    const neighbors = [];
                    // Find nodes this node depends on (outgoing)
                    outgoing.get(n.id).forEach(targetId => {{
                        const target = nodeById.get(targetId);
                        if (target && target.x !== undefined) neighbors.push(target.x);
                    }});
                    // Barycenter = average x of neighbors
                    n.barycenter = neighbors.length > 0 ? neighbors.reduce((a, b) => a + b, 0) / neighbors.length : n.x;
                }});

                nodesInLayer.sort((a, b) => a.barycenter - b.barycenter);
                positionLayer(layer);
            }}

            // Up pass: reorder based on neighbors below (lower layer = dependents)
            for (let layer = 1; layer <= maxLayer; layer++) {{
                const nodesInLayer = layerGroups.get(layer);
                if (!nodesInLayer || nodesInLayer.length < 2) continue;

                nodesInLayer.forEach(n => {{
                    const neighbors = [];
                    // Find nodes that depend on this node (incoming)
                    incoming.get(n.id).forEach(sourceId => {{
                        const source = nodeById.get(sourceId);
                        if (source && source.x !== undefined) neighbors.push(source.x);
                    }});
                    n.barycenter = neighbors.length > 0 ? neighbors.reduce((a, b) => a + b, 0) / neighbors.length : n.x;
                }});

                nodesInLayer.sort((a, b) => a.barycenter - b.barycenter);
                positionLayer(layer);
            }}
        }}

        // Draw curved links (Doxygen style with smooth bezier curves)
        const link = g.append('g')
            .selectAll('path')
            .data(links)
            .join('path')
            .attr('fill', 'none')
            .attr('stroke', '#606060')
            .attr('stroke-width', 1)
            .attr('marker-end', 'url(#doxygen-arrow)')
            .attr('d', d => {{
                const src = nodeById.get(typeof d.source === 'object' ? d.source.id : d.source);
                const tgt = nodeById.get(typeof d.target === 'object' ? d.target.id : d.target);
                if (!src || !tgt) return '';

                const srcY = src.y + nodeHeight / 2;
                const tgtY = tgt.y - nodeHeight / 2;

                // Smooth cubic bezier curve for easy-to-follow relationships
                const controlOffset = Math.abs(tgtY - srcY) * 0.5;
                return `M${{src.x}},${{srcY}} C${{src.x}},${{srcY + controlOffset}} ${{tgt.x}},${{tgtY - controlOffset}} ${{tgt.x}},${{tgtY}}`;
            }});

        // Draw nodes (Doxygen-style rectangles)
        const node = g.append('g')
            .selectAll('g')
            .data(nodes)
            .join('g')
            .attr('class', 'doxygen-node')
            .attr('transform', d => `translate(${{d.x}},${{d.y}})`)
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        // Rectangle background
        node.append('rect')
            .attr('x', d => -d.nodeWidth / 2)
            .attr('y', -nodeHeight / 2)
            .attr('width', d => d.nodeWidth)
            .attr('height', nodeHeight)
            .attr('rx', 2)
            .attr('ry', 2)
            .attr('fill', d => d.isHeader ? 'url(#headerGradient)' : 'url(#sourceGradient)')
            .attr('stroke', d => d.isHeader ? '#84b0c7' : '#c0a000')
            .attr('stroke-width', 1);

        // File name text
        node.append('text')
            .attr('class', 'node-label')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('font-family', 'Consolas, Monaco, monospace')
            .attr('font-size', '11px')
            .attr('fill', '#333')
            .text(d => d.name);

        // Tooltip styling update
        const tooltip = document.getElementById('tooltip');
        tooltip.style.background = '#ffffcc';
        tooltip.style.border = '1px solid #333';
        tooltip.style.color = '#333';
        tooltip.style.fontFamily = 'Segoe UI, Tahoma, sans-serif';
        tooltip.style.fontSize = '12px';
        tooltip.style.borderRadius = '3px';
        tooltip.style.boxShadow = '2px 2px 5px rgba(0,0,0,0.2)';

        node.on('mouseover', (event, d) => {{
            // Highlight this node
            d3.select(event.currentTarget).select('rect')
                .attr('fill', 'url(#highlightGradient)')
                .attr('stroke', '#228b22')
                .attr('stroke-width', 2);

            tooltip.innerHTML = `
                <strong>${{d.path}}</strong><br>
                <span style="color:#666">Depth: ${{d.graphDepth}} | Lines: ${{d.lines}}</span><br>
                <span style="color:#006400">Includes: ${{d.fanOut}} files</span><br>
                <span style="color:#00008b">Included by: ${{d.fanIn}} files</span>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';

            // Highlight connected nodes and links
            const connectedNodes = new Set([d.id]);
            outgoing.get(d.id).forEach(id => connectedNodes.add(id));
            incoming.get(d.id).forEach(id => connectedNodes.add(id));

            node.style('opacity', n => connectedNodes.has(n.id) ? 1 : 0.3);
            link.style('opacity', l => {{
                const srcId = typeof l.source === 'object' ? l.source.id : l.source;
                const tgtId = typeof l.target === 'object' ? l.target.id : l.target;
                return (srcId === d.id || tgtId === d.id) ? 1 : 0.15;
            }})
            .attr('stroke', l => {{
                const srcId = typeof l.source === 'object' ? l.source.id : l.source;
                const tgtId = typeof l.target === 'object' ? l.target.id : l.target;
                if (srcId === d.id) return '#006400';  // outgoing = green
                if (tgtId === d.id) return '#00008b';  // incoming = blue
                return '#606060';
            }})
            .attr('stroke-width', l => {{
                const srcId = typeof l.source === 'object' ? l.source.id : l.source;
                const tgtId = typeof l.target === 'object' ? l.target.id : l.target;
                return (srcId === d.id || tgtId === d.id) ? 2 : 1;
            }});
        }})
        .on('mouseout', (event, d) => {{
            // Reset node style
            d3.select(event.currentTarget).select('rect')
                .attr('fill', d.isHeader ? 'url(#headerGradient)' : 'url(#sourceGradient)')
                .attr('stroke', d.isHeader ? '#84b0c7' : '#c0a000')
                .attr('stroke-width', 1);

            tooltip.style.display = 'none';

            // Reset all
            node.style('opacity', currentFilter ? (n => n.directory === currentFilter ? 1 : 0.3) : 1);
            link.style('opacity', currentFilter ? 0.5 : 1)
                .attr('stroke', '#606060')
                .attr('stroke-width', 1);
        }});

        function dragstarted(event) {{
            event.subject.dragging = true;
        }}

        function dragged(event) {{
            event.subject.x = event.x;
            event.subject.y = event.y;
            d3.select(event.sourceEvent.target.parentNode)
                .attr('transform', `translate(${{event.x}},${{event.y}})`);

            // Update connected links with curved bezier paths
            link.attr('d', d => {{
                const src = nodeById.get(typeof d.source === 'object' ? d.source.id : d.source);
                const tgt = nodeById.get(typeof d.target === 'object' ? d.target.id : d.target);
                if (!src || !tgt) return '';

                const srcY = src.y + nodeHeight / 2;
                const tgtY = tgt.y - nodeHeight / 2;

                // Smooth cubic bezier curve
                const controlOffset = Math.abs(tgtY - srcY) * 0.5;
                return `M${{src.x}},${{srcY}} C${{src.x}},${{srcY + controlOffset}} ${{tgt.x}},${{tgtY - controlOffset}} ${{tgt.x}},${{tgtY}}`;
            }});
        }}

        function dragended(event) {{
            event.subject.dragging = false;
        }}

        // Filter by directory
        document.getElementById('dirFilter').addEventListener('change', (e) => {{
            currentFilter = e.target.value;

            node.style('opacity', d => {{
                if (!currentFilter) return 1;
                return d.directory === currentFilter ? 1 : 0.3;
            }});

            link.style('opacity', l => {{
                if (!currentFilter) return 1;
                const src = nodeById.get(typeof l.source === 'object' ? l.source.id : l.source);
                const tgt = nodeById.get(typeof l.target === 'object' ? l.target.id : l.target);
                return (src.directory === currentFilter || tgt.directory === currentFilter) ? 1 : 0.15;
            }});
        }});

        // Auto-fit view to content
        setTimeout(() => {{
            const bounds = g.node().getBBox();
            const fullWidth = bounds.width + 80;
            const fullHeight = bounds.height + 80;
            const midX = bounds.x + bounds.width / 2;
            const midY = bounds.y + bounds.height / 2;

            const scale = Math.min(width / fullWidth, height / fullHeight, 0.95);
            const translateX = width / 2 - midX * scale;
            const translateY = height / 2 - midY * scale;

            svg.transition().duration(500)
                .call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(scale));
        }}, 100);

        // Tab switching
        function showTab(tabId) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }}

        // =================================================================
        // Clean Architecture Module Graph (D3.js with CA layer grouping)
        // =================================================================
        const caContainer = document.getElementById('ca-graph');
        const caWidth = caContainer ? caContainer.clientWidth : 800;
        const caHeight = caContainer ? caContainer.clientHeight : 500;

        if (caContainer && caLayers.length > 0) {{
            // Group files by directory and determine CA layer
            const dirFiles = {{}};
            nodes.forEach(n => {{
                const dir = n.directory || '.';
                if (!dirFiles[dir]) dirFiles[dir] = [];
                dirFiles[dir].push(n);
            }});

            // Determine dominant CA layer for each directory
            const dirCALayers = {{}};
            Object.keys(dirFiles).forEach(dir => {{
                const layerCounts = {{}};
                dirFiles[dir].forEach(n => {{
                    layerCounts[n.layer] = (layerCounts[n.layer] || 0) + 1;
                }});
                let maxCount = 0, dominantLayer = 'Application';
                Object.entries(layerCounts).forEach(([layer, count]) => {{
                    if (count > maxCount) {{ maxCount = count; dominantLayer = layer; }}
                }});
                dirCALayers[dir] = dominantLayer;
            }});

            // Build module nodes with CA layer info
            const caModules = Object.keys(dirFiles).map(dir => ({{
                id: dir,
                name: dir,
                fileCount: dirFiles[dir].length,
                caLayer: dirCALayers[dir],
                color: caLayers.find(l => l.name === dirCALayers[dir])?.color || '#888'
            }}));

            // Build module links from file dependencies
            const caModuleLinks = [];
            const seenLinks = new Set();
            Object.keys(dirFiles).forEach(srcDir => {{
                dirFiles[srcDir].forEach(srcNode => {{
                    links.forEach(link => {{
                        if (link.source === srcNode.id) {{
                            const tgtNode = nodes.find(n => n.id === link.target);
                            if (tgtNode) {{
                                const tgtDir = tgtNode.directory || '.';
                                if (srcDir !== tgtDir) {{
                                    const key = srcDir + '|' + tgtDir;
                                    if (!seenLinks.has(key)) {{
                                        seenLinks.add(key);
                                        caModuleLinks.push({{ source: srcDir, target: tgtDir, value: 1 }});
                                    }}
                                }}
                            }}
                        }}
                    }});
                }});
            }});

            // Create SVG
            const caSvg = d3.select('#ca-graph')
                .append('svg')
                .attr('width', caWidth)
                .attr('height', caHeight);

            // Gradients for each layer
            const caDefs = caSvg.append('defs');
            caLayers.forEach(l => {{
                const grad = caDefs.append('linearGradient')
                    .attr('id', 'caGrad-' + l.name)
                    .attr('x1', '0%').attr('y1', '0%')
                    .attr('x2', '0%').attr('y2', '100%');
                grad.append('stop').attr('offset', '0%').attr('stop-color', l.color).attr('stop-opacity', 0.3);
                grad.append('stop').attr('offset', '100%').attr('stop-color', l.color).attr('stop-opacity', 0.5);
            }});

            // Arrow marker
            caDefs.append('marker')
                .attr('id', 'ca-arrow')
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 10).attr('refY', 0)
                .attr('markerWidth', 8).attr('markerHeight', 8)
                .attr('orient', 'auto')
                .append('path').attr('d', 'M0,-4L10,0L0,4').attr('fill', '#606060');

            const caG = caSvg.append('g');

            // Zoom
            const caZoom = d3.zoom()
                .scaleExtent([0.3, 3])
                .on('zoom', (event) => caG.attr('transform', event.transform));
            caSvg.call(caZoom);

            // Build adjacency for layer calculation
            const caOutgoing = new Map();
            const caIncoming = new Map();
            caModules.forEach(n => {{
                caOutgoing.set(n.id, new Set());
                caIncoming.set(n.id, new Set());
            }});
            caModuleLinks.forEach(l => {{
                if (caOutgoing.has(l.source)) caOutgoing.get(l.source).add(l.target);
                if (caIncoming.has(l.target)) caIncoming.get(l.target).add(l.source);
            }});

            // Group modules by CA layer
            const modulesByCALayer = {{}};
            caLayers.forEach(l => modulesByCALayer[l.name] = []);
            caModules.forEach(m => {{
                if (modulesByCALayer[m.caLayer]) modulesByCALayer[m.caLayer].push(m);
            }});

            // Layout constants
            const nodeHeight = 30;
            const nodePadding = 12;
            const layerSpacing = 100;
            const nodeSpacingH = 25;

            // Calculate node widths
            const caTempText = caSvg.append('text')
                .attr('font-family', 'Consolas, Monaco, monospace')
                .attr('font-size', '12px');

            caModules.forEach(n => {{
                caTempText.text(n.name);
                n.textWidth = caTempText.node().getComputedTextLength();
                n.nodeWidth = Math.max(90, n.textWidth + nodePadding * 2);
            }});
            caTempText.remove();

            // Position modules by CA layer:
            // Presentation(top-left) -> Application(mid-left) -> Core(bottom)
            //                           Infrastructure(mid-right) -> Core(bottom)
            const layerOrder = ['Presentation', 'Application', 'Infrastructure', 'Core'];
            const startY = 60;

            layerOrder.forEach(layerName => {{
                const nodesInLayer = modulesByCALayer[layerName] || [];
                let y, xOffset, availWidth;

                if (layerName === 'Presentation') {{
                    // Top-left, only above Application
                    y = startY;
                    xOffset = 0;
                    availWidth = caWidth / 2 - 20;
                }} else if (layerName === 'Application') {{
                    // Middle-left, below Presentation
                    y = startY + layerSpacing;
                    xOffset = 0;
                    availWidth = caWidth / 2 - 20;
                }} else if (layerName === 'Infrastructure') {{
                    // Middle-right, same row as Application (no Presentation above)
                    y = startY + layerSpacing;
                    xOffset = caWidth / 2;
                    availWidth = caWidth / 2 - 20;
                }} else {{
                    // Core at bottom, full width
                    y = startY + 2 * layerSpacing;
                    xOffset = 0;
                    availWidth = caWidth;
                }}

                const totalWidth = nodesInLayer.reduce((sum, n) => sum + n.nodeWidth, 0) + (nodesInLayer.length - 1) * nodeSpacingH;
                let x = xOffset + (availWidth - totalWidth) / 2;

                nodesInLayer.forEach(n => {{
                    n.x = x + n.nodeWidth / 2;
                    n.y = y;
                    x += n.nodeWidth + nodeSpacingH;
                }});
            }});

            // Draw layer backgrounds
            const layerBgData = [
                {{ name: 'Presentation', y: startY - 30, height: 70, x: 10, width: caWidth / 2 - 20 }},
                {{ name: 'Application', y: startY + layerSpacing - 30, height: 70, x: 10, width: caWidth / 2 - 20 }},
                {{ name: 'Infrastructure', y: startY + layerSpacing - 30, height: 70, x: caWidth / 2 + 10, width: caWidth / 2 - 20 }},
                {{ name: 'Core', y: startY + 2 * layerSpacing - 30, height: 70, x: 10, width: caWidth - 20 }}
            ];

            layerBgData.forEach(bg => {{
                const layer = caLayers.find(l => l.name === bg.name);
                if (!layer) return;
                caG.append('rect')
                    .attr('x', bg.x).attr('y', bg.y)
                    .attr('width', bg.width).attr('height', bg.height)
                    .attr('rx', 6)
                    .attr('fill', layer.color).attr('fill-opacity', 0.08)
                    .attr('stroke', layer.color).attr('stroke-width', 1.5)
                    .attr('stroke-dasharray', '5,3');
                caG.append('text')
                    .attr('x', bg.x + 10).attr('y', bg.y + 16)
                    .attr('font-size', '12px').attr('font-weight', 'bold')
                    .attr('fill', layer.color)
                    .text(layer.name);
            }});

            const caNodeById = new Map(caModules.map(n => [n.id, n]));

            // Draw curved links
            const caLink = caG.append('g')
                .selectAll('path')
                .data(caModuleLinks)
                .join('path')
                .attr('fill', 'none')
                .attr('stroke', '#606060')
                .attr('stroke-width', 1.5)
                .attr('marker-end', 'url(#ca-arrow)')
                .attr('d', d => {{
                    const src = caNodeById.get(d.source);
                    const tgt = caNodeById.get(d.target);
                    if (!src || !tgt || src.x === undefined || tgt.x === undefined) return '';

                    const srcY = src.y + nodeHeight / 2;
                    const tgtY = tgt.y - nodeHeight / 2;
                    const controlOffset = Math.abs(tgtY - srcY) * 0.5;

                    return `M${{src.x}},${{srcY}} C${{src.x}},${{srcY + controlOffset}} ${{tgt.x}},${{tgtY - controlOffset}} ${{tgt.x}},${{tgtY}}`;
                }});

            // Draw module nodes
            const caNode = caG.append('g')
                .selectAll('g')
                .data(caModules.filter(n => n.x !== undefined))
                .join('g')
                .attr('class', 'ca-node')
                .attr('transform', d => `translate(${{d.x}},${{d.y}})`);

            caNode.append('rect')
                .attr('x', d => -d.nodeWidth / 2)
                .attr('y', -nodeHeight / 2)
                .attr('width', d => d.nodeWidth)
                .attr('height', nodeHeight)
                .attr('rx', 4)
                .attr('fill', d => `url(#caGrad-${{d.caLayer}})`)
                .attr('stroke', d => d.color)
                .attr('stroke-width', 2);

            caNode.append('text')
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'central')
                .attr('font-family', 'Consolas, Monaco, monospace')
                .attr('font-size', '11px')
                .attr('fill', '#333')
                .text(d => d.name.length > 15 ? d.name.slice(0, 13) + '..' : d.name);

            // Tooltip
            caNode.on('mouseover', (event, d) => {{
                d3.select(event.currentTarget).select('rect')
                    .attr('stroke-width', 3);

                const outCount = caOutgoing.get(d.id)?.size || 0;
                const inCount = caIncoming.get(d.id)?.size || 0;
                tooltip.innerHTML = `
                    <strong>${{d.name}}</strong><br>
                    <span style="color:${{d.color}}">${{d.caLayer}}</span> (${{d.fileCount}} files)<br>
                    <span style="color:#006400">Depends on: ${{outCount}} modules</span><br>
                    <span style="color:#00008b">Used by: ${{inCount}} modules</span>
                `;
                tooltip.style.display = 'block';
                tooltip.style.left = (event.pageX + 10) + 'px';
                tooltip.style.top = (event.pageY - 10) + 'px';

                const connected = new Set([d.id]);
                caOutgoing.get(d.id)?.forEach(id => connected.add(id));
                caIncoming.get(d.id)?.forEach(id => connected.add(id));

                caNode.style('opacity', n => connected.has(n.id) ? 1 : 0.3);
                caLink.style('opacity', l => (l.source === d.id || l.target === d.id) ? 1 : 0.15)
                    .attr('stroke', l => {{
                        if (l.source === d.id) return '#006400';
                        if (l.target === d.id) return '#00008b';
                        return '#606060';
                    }});
            }})
            .on('mouseout', (event, d) => {{
                d3.select(event.currentTarget).select('rect')
                    .attr('stroke-width', 2);

                tooltip.style.display = 'none';
                caNode.style('opacity', 1);
                caLink.style('opacity', 1).attr('stroke', '#606060');
            }});

            // Auto-fit
            setTimeout(() => {{
                const bounds = caG.node().getBBox();
                const fullWidth = bounds.width + 80;
                const fullHeight = bounds.height + 80;
                const midX = bounds.x + bounds.width / 2;
                const midY = bounds.y + bounds.height / 2;

                const scale = Math.min(caWidth / fullWidth, caHeight / fullHeight, 1);
                const translateX = caWidth / 2 - midX * scale;
                const translateY = caHeight / 2 - midY * scale;

                caSvg.transition().duration(500)
                    .call(caZoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(scale));
            }}, 100);
        }} else {{
            if (caContainer) caContainer.innerHTML = '<p style="padding: 20px; color: #888;">No Clean Architecture data available.</p>';
        }}
    </script>
</body>
</html>
'''

    # Generate dynamic content
    max_included = max(c for _, c in most_included) if most_included else 1
    most_included_rows = '\n'.join(
        '<tr><td>{}</td><td>{}</td><td><div class="bar-container"><div class="bar" style="width: {}%;"></div></div></td></tr>'.format(
            f, c, int(c / max_included * 100)
        )
        for f, c in most_included
    )

    max_including = max(c for _, c in most_including) if most_including else 1
    most_including_rows = '\n'.join(
        '<tr><td>{}</td><td>{}</td><td><div class="bar-container"><div class="bar" style="width: {}%;"></div></div></td></tr>'.format(
            f, c, int(c / max_including * 100)
        )
        for f, c in most_including
    )

    dir_rows = '\n'.join(
        '<tr><td>{}</td><td>{}</td><td>{:,}</td></tr>'.format(
            d['name'], d['files'], d['lines']
        )
        for d in sorted(dir_summary, key=lambda x: -x['lines'])
    )

    dir_options = '\n'.join(
        '<option value="{0}">{0}</option>'.format(d['name'])
        for d in sorted(dir_summary, key=lambda x: x['name'])
    )

    if cycles:
        cycles_html = '<div class="cycle-warning"><h3>Circular Dependencies Detected!</h3>'
        for cycle in cycles:
            cycles_html += '<div class="cycle-path">{}</div>'.format(' &rarr; '.join(cycle))
        cycles_html += '</div>'
    else:
        cycles_html = '<div class="no-cycles">No circular dependencies detected.</div>'

    # Clean Architecture content generation
    layer_stats_html = ''
    layer_legend_html = ''
    violations_html = ''

    if ca_layers:
        # Layer statistics cards
        for layer in ca_layers:
            layer_name = layer['name']
            layer_stat = ca_layer_stats.get(layer_name, {'files': 0, 'lines': 0})
            layer_stats_html += '''
                <div class="layer-stat-card" style="background: {color}22; border-color: {color};">
                    <div class="layer-name" style="color: {color};">{name}</div>
                    <div class="layer-desc">{desc}</div>
                    <div class="layer-files">{files}</div>
                    <div style="font-size: 0.85em; color: #888;">files ({lines:,} lines)</div>
                </div>
            '''.format(
                name=layer_name,
                desc=layer['description'],
                color=layer['color'],
                files=layer_stat['files'],
                lines=layer_stat['lines'],
            )

        # Layer legend
        for layer in ca_layers:
            layer_legend_html += '''
                <div class="layer-legend-item">
                    <div class="layer-legend-color" style="background: {};"></div>
                    <span>{} ({})</span>
                </div>
            '''.format(layer['color'], layer['name'], layer['description'])

        # Violations table
        if ca_violations:
            violations_html = '''
                <h3 style="color: #F44336; margin-top: 20px;">Dependency Violations</h3>
                <p style="color: #aaa; margin-bottom: 10px;">
                    Inner layers should not depend on outer layers. These violations need attention.
                </p>
                <table class="violation-table">
                    <thead>
                        <tr>
                            <th>Source File</th>
                            <th>Source Layer</th>
                            <th></th>
                            <th>Target File</th>
                            <th>Target Layer</th>
                        </tr>
                    </thead>
                    <tbody>
            '''
            for v in ca_violations:
                src_color = next((l['color'] for l in ca_layers if l['name'] == v['source_layer']), '#888')
                tgt_color = next((l['color'] for l in ca_layers if l['name'] == v['target_layer']), '#888')
                violations_html += '''
                    <tr class="violation-row">
                        <td>{}</td>
                        <td><span style="color: {}; font-weight: bold;">{}</span></td>
                        <td style="color: #F44336;">&rarr;</td>
                        <td>{}</td>
                        <td><span style="color: {}; font-weight: bold;">{}</span></td>
                    </tr>
                '''.format(
                    v['source'], src_color, v['source_layer'],
                    v['target'], tgt_color, v['target_layer']
                )
            violations_html += '</tbody></table>'
        else:
            violations_html = '<div class="no-cycles" style="margin-top: 20px;">No Clean Architecture violations detected.</div>'

        # Warnings table (layer skipping)
        if ca_warnings:
            violations_html += '''
                <h3 style="color: #FF9800; margin-top: 20px;">Layer Skipping Warnings</h3>
                <p style="color: #aaa; margin-bottom: 10px;">
                    These dependencies skip intermediate layers. Consider routing through adjacent layers.
                </p>
                <table class="violation-table">
                    <thead>
                        <tr>
                            <th>Source File</th>
                            <th>Source Layer</th>
                            <th></th>
                            <th>Target File</th>
                            <th>Target Layer</th>
                        </tr>
                    </thead>
                    <tbody>
            '''
            for w in ca_warnings:
                src_color = next((l['color'] for l in ca_layers if l['name'] == w['source_layer']), '#888')
                tgt_color = next((l['color'] for l in ca_layers if l['name'] == w['target_layer']), '#888')
                violations_html += '''
                    <tr style="background: rgba(255, 152, 0, 0.1);">
                        <td>{}</td>
                        <td><span style="color: {}; font-weight: bold;">{}</span></td>
                        <td style="color: #FF9800;">&rarr;</td>
                        <td>{}</td>
                        <td><span style="color: {}; font-weight: bold;">{}</span></td>
                    </tr>
                '''.format(
                    w['source'], src_color, w['source_layer'],
                    w['target'], tgt_color, w['target_layer']
                )
            violations_html += '</tbody></table>'

    # Format HTML
    html = html.format(
        project_path=scanner.root_path,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        total_files=stats['total_files'],
        header_files=stats['header_files'],
        source_files=stats['source_files'],
        total_lines=stats['total_lines'],
        total_deps=stats['total_dependencies'],
        cycle_count=len(cycles),
        nodes_json=json.dumps(nodes),
        links_json=json.dumps(links),
        dir_deps_json=json.dumps(dir_deps),
        ca_layers_json=json.dumps(ca_layers),
        ca_violations_json=json.dumps(ca_violations),
        most_included_rows=most_included_rows,
        most_including_rows=most_including_rows,
        dir_rows=dir_rows,
        dir_options=dir_options,
        cycles_html=cycles_html,
        violation_count=len(ca_violations),
        layer_stats_html=layer_stats_html,
        layer_legend_html=layer_legend_html,
        violations_html=violations_html,
        python_version='{}.{}.{}'.format(*sys.version_info[:3]),
    )

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze C/C++ header dependencies and generate HTML visualization.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
  %(prog)s /path/to/project

  First run: generates ca_layers.json with auto-detected layers
  Edit ca_layers.json to match your design concept
  Run again: checks for Clean Architecture violations

Clean Architecture Layers (outermost to innermost):
  Presentation   - Entry points, UI, drivers (outermost)
  Application    - Use cases, orchestration
  Core           - Business logic, interfaces
  Infrastructure - External services, utilities (innermost)

  Layers are auto-detected based on dependency analysis.
  Violations occur when inner layers depend on outer layers.

Output:
  The tool generates an interactive HTML file with:
  - Dependency graph (Doxygen-style with curved lines)
  - Clean Architecture radial visualization
  - Layer violation detection and reporting
  - Circular dependency detection
  - Most included/including files analysis
        '''
    )

    parser.add_argument(
        'project_path',
        help='Path to the project root directory'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 2.0.0 (Python {}.{}.{})'.format(*sys.version_info[:3])
    )

    args = parser.parse_args()

    # Validate project path
    if not os.path.isdir(args.project_path):
        print("Error: '{}' is not a valid directory".format(args.project_path))
        sys.exit(1)

    exclude_dirs = set(DEFAULT_EXCLUDES)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'ca_layers.json')

    print("C/C++ Dependency Analyzer with Clean Architecture Analysis")
    print("=" * 60)
    print("Project: {}".format(os.path.abspath(args.project_path)))
    print("Excluding: {}".format(', '.join(sorted(exclude_dirs))))
    print()

    # Scan
    print("Scanning files...")

    scanner = DependencyScanner(
        args.project_path,
        exclude_dirs=exclude_dirs
    )
    scanner.scan()

    stats = scanner.get_stats()

    print("Found {} files ({} headers, {} sources)".format(
        stats['total_files'],
        stats['header_files'],
        stats['source_files']
    ))
    print("Total lines: {:,}".format(stats['total_lines']))
    print("Dependencies: {}".format(stats['total_dependencies']))

    if stats['unresolved_includes'] > 0:
        print("Unresolved includes: {}".format(stats['unresolved_includes']))

    cycles = scanner.find_cycles()
    if cycles:
        print("\nWarning: {} circular dependencies found!".format(len(cycles)))

    # Clean Architecture Analysis
    print()

    # Check if config exists
    config_exists = os.path.exists(config_path)

    if config_exists:
        print("Using existing config: {}".format(config_path))
        ca_analyzer = CleanArchAnalyzer(scanner, config_path=config_path)
    else:
        print("Auto-detecting Clean Architecture layers...")
        ca_analyzer = CleanArchAnalyzer(scanner)

    ca_analyzer.analyze()

    # Generate config if it doesn't exist
    if not config_exists:
        ca_analyzer.generate_config_template(config_path)
        print("Generated config: {}".format(config_path))
        print("  Edit this file to customize layer assignments, then run again.")

    layer_stats = ca_analyzer.get_layer_stats()
    print("Layer distribution:")
    for layer_name, layer_stat in layer_stats.items():
        print("  {}: {} files ({:,} lines)".format(
            layer_name, layer_stat['files'], layer_stat['lines']
        ))

    if ca_analyzer.violations:
        print("\nError: {} Clean Architecture violations found!".format(
            len(ca_analyzer.violations)
        ))
        violation_summary = ca_analyzer.get_violation_summary()
        for violation_type, count in violation_summary.items():
            print("  {} : {} occurrences".format(violation_type, count))
    else:
        print("\nNo Clean Architecture violations detected.")

    if ca_analyzer.warnings:
        print("\nWarning: {} layer skipping instances found:".format(
            len(ca_analyzer.warnings)
        ))
        warning_summary = ca_analyzer.get_warning_summary()
        for warning_type, count in warning_summary.items():
            print("  {} : {} occurrences (skips layer)".format(warning_type, count))
        print("  Details:")
        for w in ca_analyzer.warnings:
            print("    {} -> {}".format(w['source'], w['target']))

    print()
    print("Generating HTML report...")

    # Generate report (output to analyzer directory)
    output_file = os.path.join(script_dir, 'dep_report.html')
    output_path = generate_html_report(scanner, output_file, ca_analyzer)

    print("Report saved to: {}".format(os.path.abspath(output_path)))
    print()
    print("Open the HTML file in a browser to view the interactive visualization.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
