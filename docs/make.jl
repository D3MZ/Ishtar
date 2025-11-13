using Ishtar
using Documenter

DocMeta.setdocmeta!(Ishtar, :DocTestSetup, :(using Ishtar); recursive=true)

makedocs(;
    modules=[Ishtar],
    authors="Demetrius Michael <arrrwalktheplank@gmail.com> and contributors",
    sitename="Ishtar.jl",
    format=Documenter.HTML(;
        canonical="https://D3MZ.github.io/Ishtar.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/D3MZ/Ishtar.jl",
    devbranch="main",
)
