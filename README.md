# BlockSparseMatrices

<p align="center">
  <img src="docs/src/assets/logo.svg" alt="Description" width="150"/>
</p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://djukic14.github.io/BlockSparseMatrices.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://djukic14.github.io/BlockSparseMatrices.jl/dev/)
[![Build Status](https://github.com/djukic14/BlockSparseMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/djukic14/BlockSparseMatrices.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/djukic14/BlockSparseMatrices.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/djukic14/BlockSparseMatrices.jl)

## Introduction

BlockSparseMatrices.jl provides a representation for sparse matrices that are composed of a limited number of (dense) blocks. It also includes specialized algorithms for symmetric block‑sparse matrices, storing only the necessary half of the off‑diagonal blocks.

## Installation

The package can be installed by

```@julia
pkg> add BlockSparseMatrices 
```
