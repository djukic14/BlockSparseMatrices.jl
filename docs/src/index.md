# BlockSparseMatrices.jl  

**BlockSparseMatrices.jl** provides a representation for sparse matrices that are composed of a limited number of (dense) blocks.  
It also includes specialized algorithms for symmetric block‑sparse matrices, storing only the necessary half of the off‑diagonal blocks.  

## Key Features  

- **Block‑sparse storage** – the matrix is built from a small set of (dense) sub‑blocks.  
- **Symmetric support** – for symmetric block‑sparse matrices only the lower (or upper) triangular block‑structure is kept, reducing memory usage.  
- **Multithreaded matrix–vector multiplication** – leverages  [OhMyThreads](https://github.com/JuliaFolds2/OhMyThreads.jl) and [GraphsColoring](https://github.com/JuliaGraphs/GraphsColoring.jl) for safe parallelism.  

### Implemented Operations  

| Operation | Description |
|-----------|-------------|
| `*` (matrix‑vector product) | Fast, multithreaded multiplication. |
| `transpose` / `adjoint` | Returns the (adjoint) transpose of a block‑sparse matrix. |
| Visualization | Visual inspection via [UnicodePlots](https://github.com/JuliaPlots/UnicodePlots.jl). |
| `sparse` | Convert to a standard `SparseMatrixCSC` from [SparseArrays](https://github.com/JuliaSparse/SparseArrays.jl). |

## Related Packages  

If you need alternative block‑matrix representations or additional functionality, consider:

- [BlockArrays](https://github.com/JuliaArrays/BlockArrays.jl)
- [BlockMatrices](https://github.com/gajomi/BlockMatrices.jl)
- [BlockDiagonals](https://github.com/JuliaArrays/BlockDiagonals.jl)
- [BlockBandedMatrices](https://github.com/JuliaLinearAlgebra/BlockBandedMatrices.jl)
