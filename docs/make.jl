using BlockSparseMatrices
using Documenter

DocMeta.setdocmeta!(
    BlockSparseMatrices, :DocTestSetup, :(using BlockSparseMatrices); recursive=true
)

makedocs(;
    modules=[BlockSparseMatrices],
    authors="djukic14 <danijel.jukic14@gmail.com> and contributors",
    sitename="BlockSparseMatrices.jl",
    format=Documenter.HTML(;
        canonical="https://djukic14.github.io/BlockSparseMatrices.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/djukic14/BlockSparseMatrices.jl", devbranch="main")
