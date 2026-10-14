# frontier-viz

GPU-accelerated (wgpu) rendering core for meshes, fields and neural-network topology: native, browser (WebGPU) and headless.

One renderer, many hosts: the same Rust code draws a 10-million-triangle surface with a
pressure field in a desktop window, in a browser tab through WebGPU, and headless on a
cluster node to produce images. It has no dependency on a UI framework.

```
cargo run --example render_surface --features png -- surface.arrow out.png
```

## What is in this repository

- `crates/frontier-viz/`: Rust crate `frontier-viz`

## Extending

`frontier-viz` takes meshes and fields in the Arrow layout of
[frontier-format](https://github.com/DimensionLab/frontier-format), so anything a connector can
read can be drawn. Colour maps, camera models and render passes are plain Rust types; new
views (for example a model-topology view) are added as passes over a shared `GpuContext`.

## Ecosystem

| Repository | What it is |
|---|---|
| [frontier-format](https://github.com/DimensionLab/frontier-format) | Frontier data model and dataset format: the frontier-dataset specification, the plugin contract, and reference readers/writers in Rust and Python. |
| [frontier-connectors](https://github.com/DimensionLab/frontier-connectors) | Connector plugins that turn CFD/CAE solver output (OpenFOAM, VTK, Fluent, STAR-CCM+, CGNS, LS-DYNA, ...) into Frontier datasets. Write your own in an afternoon. |
| [frontier-sdk](https://github.com/DimensionLab/frontier-sdk) | Frontier Python SDK: offline-first experiment tracking for physics ML, the native Rust core, and the NVIDIA PhysicsNeMo bridge. |
| [frontier-physicsnemo](https://github.com/DimensionLab/frontier-physicsnemo) | NVIDIA PhysicsNeMo plugin for Frontier: logging, model registry, checkpoint compatibility and dataset adapters. Installable independently of the platform. |
| [frontier-hub](https://github.com/DimensionLab/frontier-hub) | Curated hub of public physics-AI datasets and benchmarks: catalog, fetchers and converters to the Frontier dataset format. |
| [frontier-editor](https://github.com/DimensionLab/frontier-editor) | Editor core and GPU text view for physics-ML configs, recipes and logs: rope document, undo tree, hunks with per-hunk approval, tree-sitter syntax, JSON-Schema diagnostics, provenance events, and a wgpu renderer that draws into the host's render pass. |
| **frontier-viz** (this repository) | GPU-accelerated (wgpu) rendering core for meshes, fields and neural-network topology: native, browser (WebGPU) and headless. |
| [foldstate](https://github.com/DimensionLab/foldstate) | Statecharts as data with an event log: a pure, deterministic kernel (state, event) -> (state, effects), replay, fork, restore and explanations. no_std + alloc, runs natively and in WebAssembly. Not specific to Frontier. |
| [frontier-zed](https://github.com/DimensionLab/frontier-zed) | Frontier for Zed: JSON schemas, snippets and the Frontier MCP context server for physics-ML training configs, RunSpecs, data-prep recipes, pipelines, hub entries and plugin manifests. A Zed extension (extension.toml plus a small wasm32-wasip2 crate), installable as a dev extension today and submitted to the Zed extension registry later. |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under [Apache-2.0](LICENSE).
