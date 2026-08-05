using Pkg

repo = get(ENV, "MPS_JULIQAOA_REPO_URL", "https://github.com/lanl/JuliQAOA.jl")
rev = get(ENV, "MPS_JULIQAOA_REV", "mps")

Pkg.Registry.update()
Pkg.add(PackageSpec(url=repo, rev=rev))
Pkg.precompile()
Pkg.status()
