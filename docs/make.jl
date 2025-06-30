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
    pages=[
        "Introduction" => "index.md",
        "Manual" => "manual.md",
        "Further Details" => "details.md",
        "Contributing" => "contributing.md",
        "API Reference" => "apiref.md",
    ],
)

deploydocs(;
    repo="github.com/djukic14/BlockSparseMatrices.jl",
    target="build",
    devbranch="main",
    push_preview=true,
    forcepush=true,
    versions=["stable" => "v^", "dev" => "dev"],
)
