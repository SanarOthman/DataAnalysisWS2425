### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ b436137f-e857-4829-8031-e1ac0bf87090
begin
    import Pkg
    
    Pkg.activate(joinpath(@__DIR__, ".."))
    
    Pkg.instantiate()
   
    using DataAnalysisWS2425, Random, Plots
end

# ╔═╡ eab9c1e0-a2d9-11ef-2249-659abd3dcc24
begin
	using DataFrames
	using UnROOT
	using XRootD
	using Statistics
	using FHist
	using StatsBase
	using LsqFit
	using Parameters
	using QuadGK
	using Optim
	 
end

# ╔═╡ 6f4a9eb2-bc6e-4b7b-9a13-88ec6da9d5fa
md"""
# Sanar Othman_Sheet 2
"""

# ╔═╡ 4165cb37-9a3b-44e8-8c43-15f21b69bfbd
md"""
## Exercise 1
"""

# ╔═╡ b814b5d0-3b3b-4648-8e85-e6794780975a
begin
	rf = ("C:\\Users\\sanar\\DataAnalysisWS2425\\Exercises2425\\sheet-02_sample-03(1).root")
	#rf = ROOTFile("sheet-02_sample-03(1).root")
	tree = LazyTree(rf,"t") |> DataFrame
end 

# ╔═╡ 37db193c-6701-4126-b667-8d1e31a47807
begin
	FE = filter(row -> row.L_M >= 1108 && row.L_M <= 1123 && row.Xi_M >= 1312 && row.Xi_M <= 1330, tree)
end

# ╔═╡ 870a335b-61a3-4ab9-9f65-87d6b1eb7ed6
begin

	   h1 = Hist1D(binedges=range(2200, 2560, length=91))
    push!.(h1, tree.Lc_M)  

    h2 = Hist1D(binedges=range(2200, 2560, length=91))  
    push!.(h2, FE.Lc_M)  


    # Before selection
    plot1 = plot(h1, title="Lc_M Before Selection", xlab="Lc_M (MeV)", ylab="Counts", label="Before Selection", legend=:right, fill=0, alpha=0.4, lw=2, lc=:yellow)
end


# ╔═╡ 57bbc194-aeba-4866-a52a-fb52e4bdfe8a
    # after selection
    plot2 = plot(h2, title="Lc_M After Selection", xlab="Lc_M (MeV)", ylab="Counts", label="After Selection", legend=:right, lw=2, lc=:red)

# ╔═╡ 0d438796-31f3-4fb2-8f62-49081cf27e01
md"""
## Exercise 2
"""

# ╔═╡ a24e5cb8-e771-48c5-ab80-82a959286a6b
  data = tree.Lc_M

# ╔═╡ 0bfddc9d-6398-4aee-96dc-6e2f0a23b4f3
begin
	 n_bins = 90
 support = (2200,2560)
    function signal_func(x, pars)
        @unpack μ, σ, a = pars
        gaussian_scaled.(x; μ, σ, a)
		
    end
    function background_func(x, pars)
        @unpack x0, x1 = pars
        polynomial_scaled.(x; coeffs =[x0, x1])
    end
    
	model_func(x, pars) = signal_func(x, pars) + background_func(x, pars)
	
	p0 = (; μ = 2286.0, σ = 10.0, x0 = -4000, x1 = 1.8, a = 800)
	@assert model_func(2.2, p0) isa Number

	x_data = bincenters(h2)
	y_data = h2.bincounts

	p_values= [p0.μ, p0.σ, p0.a, p0.x0, p0.x1]

	
	fit_result = curve_fit((x, p0) -> model_func(x, (; μ = p0[1], σ = p0[2], a =p0[3], x0 = p0[4], x1 = p0[5])), x_data, y_data, p_values)

	bfp = fit_result.param

	plot(h2; seriestype = :stepbins, xlabel = "Lc_M (MeV)", ylabel = "Frequency", 
label = "Data", xlim = support, lc=:black)

	plot!(x -> model_func(x, (; μ = bfp[1], σ = bfp[2], 
a = bfp[3], x0 = bfp[4], x1 = bfp[5])), support[1]:1:support[end], lw = 2, lc = :red, ls = :dash, label = "Fitted Model")
	
end

# ╔═╡ 98a34a82-0c9b-4081-a29a-a4981f42e235
md"""
The first order polynomial is chosen because it sufficiently captures the smooth and slow background variation and avoids overfitting. 
"""

# ╔═╡ d0ad66d4-7dad-4a84-9cf5-498f3c32b09c
md"""
## Exercise 3
"""

# ╔═╡ 12d610b2-3e73-4702-8efa-79ca050607b2
begin
    # Calculate pull values
    y_model = model_func(x_data, (; μ = bfp[1], σ = bfp[2], a = bfp[3], x0 = bfp[4], x1 = bfp[5]))
    
    error_data = sqrt.(y_data)

    pull_values = (y_data .- y_model) ./ error_data

    plot(x_data, pull_values; seriestype=:line, xlabel="Lc_M (MeV)", ylabel="Pull", 
         label="Pull Values", title="Pull Values vs Lc_M", legend=false)
    hline!([0], lw=2, lc=:black) 
end


# ╔═╡ 2a5c72f8-8c85-4e30-b1e8-3e83d2131bbf
md""" 
##### The fit is generally good, but the cluster around LcM≈2400 MeV suggests there may be an additional structure or a slight mismatch in the model.
"""

# ╔═╡ 20154189-7998-421d-a4fb-d1b1115154a8
histogram(pull_values; bins=20, xlabel="Pull", ylabel="Frequency", 
              title="Histogram of Pull Values", label="", lc=:blue)

# ╔═╡ 4a5547a3-9294-4770-ad14-a24794d41a7f
md"""
##### The histogram must likely resemble a STD that is centered at zero (μ = 0) and std is close to one (σ = 1) and there are no skewness and/or long tails.
"""

# ╔═╡ 46cb351a-2b56-4f9e-bef4-930bf0108adf
md"""
## Exercise 4
"""

# ╔═╡ a5ccf94f-970d-4690-8708-1f9cf06220f5
bfp[1]

# ╔═╡ 4e18220d-e5f8-4f8a-8845-af41112c208d
begin 
	estimated_mass = bfp[1]
	known_mass = 2286.46

	mass_difference = estimated_mass - known_mass

	println("Estimated mass of Λ_c^+: $estimated_mass MeV/c^2")
	println("Known mass of Λ_c^+: $known_mass MeV/c^2")
	println("Difference: $mass_difference MeV/c^2")
end

# ╔═╡ Cell order:
# ╟─6f4a9eb2-bc6e-4b7b-9a13-88ec6da9d5fa
# ╠═b436137f-e857-4829-8031-e1ac0bf87090
# ╠═eab9c1e0-a2d9-11ef-2249-659abd3dcc24
# ╟─4165cb37-9a3b-44e8-8c43-15f21b69bfbd
# ╠═b814b5d0-3b3b-4648-8e85-e6794780975a
# ╠═37db193c-6701-4126-b667-8d1e31a47807
# ╠═870a335b-61a3-4ab9-9f65-87d6b1eb7ed6
# ╠═57bbc194-aeba-4866-a52a-fb52e4bdfe8a
# ╟─0d438796-31f3-4fb2-8f62-49081cf27e01
# ╠═a24e5cb8-e771-48c5-ab80-82a959286a6b
# ╠═0bfddc9d-6398-4aee-96dc-6e2f0a23b4f3
# ╟─98a34a82-0c9b-4081-a29a-a4981f42e235
# ╟─d0ad66d4-7dad-4a84-9cf5-498f3c32b09c
# ╠═12d610b2-3e73-4702-8efa-79ca050607b2
# ╟─2a5c72f8-8c85-4e30-b1e8-3e83d2131bbf
# ╠═20154189-7998-421d-a4fb-d1b1115154a8
# ╟─4a5547a3-9294-4770-ad14-a24794d41a7f
# ╟─46cb351a-2b56-4f9e-bef4-930bf0108adf
# ╠═a5ccf94f-970d-4690-8708-1f9cf06220f5
# ╠═4e18220d-e5f8-4f8a-8845-af41112c208d
