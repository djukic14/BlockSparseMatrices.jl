
# Further Details

## Matrix-Vector Product
For the multithreaded matrix-vector product

$$\bm y = \bm A \bm x,$$

where $\bm A$ is a `BlockSparseMatrix` or a `SymmetricBlockMatrix`, $\bm x$ a random vector, and $\bm y$ the solution vector, we use a coloring scheme to allow a multithreaded matrix-vector product without unnecessary allocations.
Each block is assigned to one color such that all blocks of the same color do not share common row or column indices. 
This allows to evaluate all matrix-vector products within one color parallel without allocating multiple solution vectors $\bm y$. 

The coloring scheme works best for large numbers of dense blocks.

The row and column indices of blocks should not be overlapping. Only unique sets of row and column indices are supported as well as row and column indices that are subsets of the indices of other blocks. 

### Example
Example of the coloring scheme using four groups. Each color can be evaluated using multiple threads with a single solution vector 

```@raw html


<div style="display: flex; gap: 20px; justify-content: space-between;">

  <figure style="width: 49%; margin: 0;">
    <img src="../assets/nocolor.svg" style="width: 100%; height: auto;" />
    <figcaption style="text-align: center; margin-top: 5px;">Figure 1: Dense blocks in a block sparse matrix. </figcaption>
  </figure>

  <figure style="width: 49%; margin: 0;">
    <img src="../assets/logo.svg" style="width: 100%; height: auto;" />
    <figcaption style="text-align: center; margin-top: 5px;">Figure 2: Colored dense blocks.</figcaption>
  </figure>

</div>
```
