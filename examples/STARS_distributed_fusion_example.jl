using Dates
using Rasters
using STARSDataFusion
# using STARSDataFusion.STARSDataFusion
using LinearAlgebra
using Statistics
using Plots
using Distributed

addprocs(4)
@everywhere using STARSDataFusion
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

coarse_image_filenames = readdir("salton_sea_coarse_NIR", join=true)
coarse_images_rr = Raster.(coarse_image_filenames, missingval=NaN)

fine_image_filenames = readdir("salton_sea_fine_NIR", join=true)
fine_images_rr = Raster.(fine_image_filenames, missingval=NaN)

nr, nc = size(coarse_images_rr[1])
nfr, nfc = size(fine_images_rr[1])
nt = size(coarse_images_rr)[1]
coarse_images = Raster(fill(NaN, nr, nc, nt), dims = (coarse_images_rr[1].dims[1:2]...,Band(1:nt)),missingval=coarse_images_rr[1].missingval)
fine_images = Raster(fill(NaN, nfr, nfc, nt), dims = (fine_images_rr[1].dims[1:2]...,Band(1:nt)),missingval=fine_images_rr[1].missingval)

for i in 1:nt
    coarse_images[:,:,i] = coarse_images_rr[i]
    fine_images[:,:,i] = fine_images_rr[i]
end

fine_images[:,:,32] .= NaN
fine_images[:,:,57] .= NaN
fine_images[:,:,73] .= NaN
fine_images[:,:,89] .= NaN;

#####

hls_times = findall(mean(isnan.(fine_images[:,:,20:41]), dims=(1,2))[:] .< 0.3)
viirs_times = findall(mean(isnan.(coarse_images[:,:,20:41]), dims=(1,2))[:] .< 0.3)

hls_array = Array{Float64}(fine_images[:,:,20:41])[:,:,hls_times]
viirs_array = Array{Float64}(coarse_images[:,:,20:41])[:,:,viirs_times]

hls_ndims = collect(size(fine_images)[1:2])
viirs_ndims = collect(size(coarse_images)[1:2])

## instrument origins and cell sizes
hls_origin = get_centroid_origin_raster(fine_images)
viirs_origin = get_centroid_origin_raster(coarse_images)

hls_csize = collect(cell_size(fine_images))
viirs_csize = collect(cell_size(coarse_images))

hls_geodata = STARSInstrumentGeoData(hls_origin, hls_csize, hls_ndims, 0, hls_times)
viirs_geodata = STARSInstrumentGeoData(viirs_origin, viirs_csize, viirs_ndims, 2, viirs_times)
window_geodata = STARSInstrumentGeoData(viirs_origin, viirs_csize, viirs_ndims, 0, viirs_times)
target_geodata = STARSInstrumentGeoData(hls_origin, hls_csize, hls_ndims, 0, viirs_times)

hls_data = STARSInstrumentData(hls_array, 0.0, 1e-6, false, nothing, abs.(hls_csize), hls_times, [1. 1.])
viirs_data = STARSInstrumentData(viirs_array, 0.0, 1e-5, false, nothing, abs.(viirs_csize), viirs_times, [1. 1.])

### parameter estimation
n_eff = compute_n_eff(Int(round(viirs_csize[1]/hls_csize[1])),2.0,smoothness=0.5) ## Matern: range = 200m, smoothness = 1.5
sp_var = fast_var_est(coarse_images[:,:,15:27], n_eff_agg = n_eff);

sp_rs = resample(log.(sqrt.(sp_var[:,:,1])); to=fine_images[:,:,1], size=size(fine_images)[1:2], method=:cubicspline)
sp_rs[isnan.(sp_rs)] .= nanmean(sp_rs) ### the resampling won't go outside extent

model_pars_ff = zeros((hls_ndims...,4))
model_pars_ff[:,:,1] = Array(exp.(sp_rs)) ## this model would be sqrt
model_pars_ff[:,:,2] .= 150.0
model_pars_ff[:,:,3] .= 1e-10
model_pars_ff[:,:,4] .= 0.5

@time fused_images_pmap, fused_sd_images_pmap = coarse_fine_scene_fusion_pmap(hls_data,
                            viirs_data,
                            hls_geodata, 
                            viirs_geodata,
                            viirs_geodata,
                            0.12*ones(hls_ndims...),
                            1e-2 * ones(hls_ndims...),
                            model_pars_ff;
                            nsamp = 50,
                            window_buffer = 4,
                            target_times = 1:4, 
                            spatial_mod = exp_cor,                                           
                            obs_operator = unif_weighted_obs_operator_centroid,
                            state_in_cov = false,
                            cov_wt = 0.2,
                            phi = 0.002,
                            nb_coarse = 1.0,
                            batchsize=100);

### old workflow
cov_raster_full = Raster(fill(NaN, nr, nc, 4), dims = (coarse_images.dims[1:2]...,Band(1:4)),missingval=fine_images.missingval)
cov_raster_full[:,:,1] = sp_var
cov_raster_full[:,:,2] .= 150
cov_raster_full[:,:,3] .= 1e-10
cov_raster_full[:,:,4] .= 1.5;

@time fusion_results = coarse_fine_data_fusion(
                            coarse_images[:,:,20:41], 
                            fine_images[:,:,20:41], 
                            cov_raster_full,
                            prior = nothing,
                            target_times = 1:4,
                            buffer_distance = 100.,
                            offset_ar = [1, 0.0], 
                            offset_var = [1e-6, 1e-6],    
                            default_mean = 0.12,
                            smooth = false
                        );


k=3
plot(Raster(fused_images_pmap[:,:,k], dims=fusion_results.mean.dims[1:2]), layout=(1,2), title="new flow", size=(1000,400))
plot!(fusion_results.mean[:,:,k], subplot=2, title="old flow")

plot(Raster(fused_images_pmap[:,:,k] .- fused_images_pmap[:,:,k+1], dims=fusion_results.mean.dims[1:2]), cmap=:bwr, clim=(-0.25,0.25), layout=(1,2), title="new flow", size=(1000,400))
plot!(fusion_results.mean[:,:,k] .- fusion_results.mean[:,:,k+1], cmap=:bwr, clim=(-0.25,0.25), subplot=2, title="old flow")
