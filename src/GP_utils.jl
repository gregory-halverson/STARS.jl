
using LinearAlgebra
using Statistics
using StatsBase
using Distances
import GaussianRandomFields.CovarianceFunction
import GaussianRandomFields.Matern
import GaussianRandomFields.apply
using KernelFunctions

function kernel_matrix(X::AbstractArray{T}; reg=1e-10, σ=1.0) where {T<:Real}
    Diagonal(reg * ones(size(X)[1])) + exp.(-0.5 * pairwise(SqEuclidean(), X, dims=1) ./ σ^2)
end

function matern_cor(X::AbstractArray{T}; reg=1e-10, ν=0.5, σ=1.0) where {T<:Real}
    cc = CovarianceFunction(2, Matern(σ, ν))
    Diagonal(reg * ones(size(X)[2])) + apply(cc, X, X)
end

function matern_cor_nonsym(X1::AbstractArray{T}, X2::AbstractArray{T}, pars=AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    ν = pars[3]
    k = with_lengthscale(MaternKernel(;ν=ν), σ)
    kernelmatrix(k,X1,X2, obsdim=2)
end

function matern_cor_fast(X1::AbstractArray{T}, pars=AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    ν = pars[3]
    k = with_lengthscale(MaternKernel(;ν=ν), σ)
    kernelmatrix(k,X1, obsdim=2)
end

function build_GP_var(locs, sigma, phi, nugget=1e-10)
    A = sigma .* kernel_matrix(locs, reg=nugget, σ=phi)
    return A
end

function exp_cor(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    Diagonal(reg * ones(size(X)[2])) + exp.(-pairwise(Euclidean(1e-12), X, dims=2) ./ σ)
end

function mat32_cor(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    dd = sqrt(3) .* pairwise(Euclidean(1e-12), X, dims=2) ./ σ
    reg * I + exp.(-dd).*(1.0 .+ dd)
end

function mat52_cor(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    dd = sqrt(5) .* pairwise(Euclidean(1e-12), X, dims=2) ./ σ
    Diagonal(reg * ones(size(X)[2])) + exp.(-dd).*(1.0 .+ dd .+ dd.^2)
end

function exp_corD(dd::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    reg * I + exp.(-dd ./ σ)
end

function mat32_corD(dd::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    dd = sqrt(3) .* dd ./ σ
    reg * I + exp.(-dd).*(1.0 .+ dd)
end

function mat52_corD(dd::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    dd = sqrt(5) .* dd ./ σ
    reg * I + exp.(-dd).*(1.0 .+ dd .+ dd.^2)
end

function state_cov(Xtt::AbstractArray{T}, pars::AbstractVector{T}) where T<:Real
    dd = pairwise(Euclidean(1e-12), Xtt, Xtt, dims=2) + UniformScaling(1e-10)
    phi = maximum([0.01,median(dd[:])])
    Qst = pars[1] .* exp.(-dd./phi)
    return Qst
end

