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

hls_times = findall(mean(isnan.(fine_images), dims=(1,2))[:] .< 0.3)
viirs_times = findall(mean(isnan.(coarse_images), dims=(1,2))[:] .< 0.3)

hls_array = Array{Float64}(fine_images)[:,:,hls_times]
viirs_array = Array{Float64}(coarse_images)[:,:,viirs_times]

hls_ndims = collect(size(fine_images)[1:2])
viirs_ndims = collect(size(coarse_images)[1:2])

## instrument origins and cell sizes
hls_origin = get_centroid_origin_raster(fine_images)
viirs_origin = get_centroid_origin_raster(coarse_images)

hls_csize = collect(cell_size(fine_images))
viirs_csize = collect(cell_size(coarse_images))

hls_geodata = STARSInstrumentGeoData(hls_origin, hls_csize, hls_ndims, 0, hls_times)
viirs_geodata = STARSInstrumentGeoData(viirs_origin, viirs_csize, viirs_ndims, 2, viirs_times)
target_geodata = STARSInstrumentGeoData(hls_origin, hls_csize, hls_ndims, 0, viirs_times)

scf = 5
nwindows = Int.(ceil.(hls_ndims./scf))
target_ndims = nwindows.*scf
window_geodata = STARSInstrumentGeoData(hls_origin .+ (scf-1.0)/2.0*hls_csize, scf*hls_csize, nwindows, 0, viirs_times)

### additive bias correction for viirs
fine_times = findall(mean(isnan.(fine_images), dims=(1,2))[:] .< 0.3)
fine_images_rs = resample(fine_images, to=coarse_images[:,:,1], method=:average)

errs = coarse_images[:] .- fine_images_rs[:] 
bb = nanmean(errs)
bbs = Array(mapslices(nanmean, coarse_images .- fine_images_rs, dims=3))

### dynamic bias correction
bias_field = zeros(size(coarse_images))

errs = Array(coarse_images .- fine_images_rs)
bias_field .= nanmean(errs[:])

for i in 1:length(viirs_times)
    kp = findall(.!isnan.(errs[:,:,i]))
    if length(kp) > 0
        println(i)
        bias_field[kp,i:size(bias_field,3)] .= errs[kp,i]
    end
end

hls_data = STARSInstrumentData(hls_array, 0.0, 1e-6, false, nothing, abs.(hls_csize), hls_times, [1. 1.])
viirs_data = STARSInstrumentData(viirs_array, 0.0, nanvar(errs) .+ nanmean(errs).^2, false, nothing, abs.(viirs_csize), viirs_times, [1. 1.])
viirs_bc_data = STARSInstrumentData(viirs_array .- bb, 0.0, nanvar(errs .- bb), false, nothing, abs.(viirs_csize), viirs_times, [1. 1.])
viirs_bcs_data = STARSInstrumentData(viirs_array .- bbs, 0.0, nanvar(errs .- bbs), false, nothing, abs.(viirs_csize), viirs_times, [1. 1.])
viirs_dbcs_data = STARSInstrumentData(viirs_array .- bias_field, 0.0, 1e-6, false, nothing, abs.(viirs_csize), viirs_times, [1. 1.])
viirs_cbias_data = STARSInstrumentData(viirs_array, 0.0, 1e-6, true, [1.0, 1e-6], abs.(viirs_csize), viirs_times, [1. 1.])

### parameter estimation
n_eff = compute_n_eff(Int(round(viirs_csize[1]/hls_csize[1])),2.0,smoothness=0.5) ## Matern: range = 200m, smoothness = 1.5
sp_var = fast_var_est(coarse_images, n_eff_agg = n_eff);

sp_rs = resample(log.(sqrt.(sp_var[:,:,1])); to=fine_images[:,:,1], size=size(fine_images)[1:2], method=:cubic)
sp_rs[isnan.(sp_rs)] .= nanmean(sp_rs) ### the resampling won't go outside extent

model_pars_ff = zeros((hls_ndims...,4))
model_pars_ff[:,:,1] = Array(exp.(sp_rs)) ## this model would be sqrt
model_pars_ff[:,:,2] .= viirs_csize[1]
model_pars_ff[:,:,3] .= 1e-10
model_pars_ff[:,:,4] .= 0.5

@time fused_images_pmap, fused_sd_images_pmap = coarse_fine_scene_fusion_pmap(hls_data,
                            viirs_data,
                            hls_geodata, 
                            viirs_geodata,
                            window_geodata,
                            0.12*ones(hls_ndims...),
                            1e-2 * ones(hls_ndims...),
                            model_pars_ff;
                            nsamp = 50,
                            window_buffer = 3,
                            target_times = viirs_times, 
                            spatial_mod = exp_cor,                                           
                            obs_operator = unif_weighted_obs_operator_centroid,
                            state_in_cov = false,
                            cov_wt = 0.2,
                            phi = 0.002,
                            nb_coarse = 2.0,
                            batchsize=1);

@time fused_images_bc_pmap, fused_sd_images_bc_pmap = coarse_fine_scene_fusion_pmap(hls_data,
                            viirs_bc_data,
                            hls_geodata, 
                            viirs_geodata,
                            window_geodata,
                            0.12*ones(hls_ndims...),
                            1e-2 * ones(hls_ndims...),
                            model_pars_ff;
                            nsamp = 50,
                            window_buffer = 3,
                            target_times = viirs_times, 
                            spatial_mod = exp_cor,                                           
                            obs_operator = unif_weighted_obs_operator_centroid,
                            state_in_cov = false,
                            cov_wt = 0.2,
                            phi = 0.002,
                            nb_coarse = 2.0,
                            batchsize=1);

@time fused_images_bcs_pmap, fused_sd_images_bcs_pmap = coarse_fine_scene_fusion_pmap(hls_data,
                            viirs_bcs_data,
                            hls_geodata, 
                            viirs_geodata,
                            window_geodata,
                            0.12*ones(hls_ndims...),
                            1e-2 * ones(hls_ndims...),
                            model_pars_ff;
                            nsamp = 50,
                            window_buffer = 3,
                            target_times = viirs_times, 
                            spatial_mod = exp_cor,                                           
                            obs_operator = unif_weighted_obs_operator_centroid,
                            state_in_cov = false,
                            cov_wt = 0.2,
                            phi = 0.002,
                            nb_coarse = 2.0,
                            batchsize=1);

@time fused_images_dbcs_pmap, fused_sd_images_dbcs_pmap = coarse_fine_scene_fusion_pmap(hls_data,
                            viirs_dbcs_data,
                            hls_geodata, 
                            viirs_geodata,
                            viirs_geodata,
                            0.12*ones(hls_ndims...),
                            1e-2 * ones(hls_ndims...),
                            model_pars_ff;
                            nsamp = 50,
                            window_buffer = 4,
                            target_times = viirs_times, 
                            spatial_mod = exp_cor,                                           
                            obs_operator = unif_weighted_obs_operator_centroid,
                            state_in_cov = false,
                            cov_wt = 0.2,
                            phi = 0.002,
                            ar_par = 1.0,
                            nb_coarse = 2.0,
                            batchsize=1);

# @time fused_images_dbcs_pmap2, fused_sd_images_dbcs_pmap2 = coarse_fine_scene_fusion_pmap(hls_data,
#                             viirs_dbcs_data,
#                             hls_geodata, 
#                             viirs_geodata,
#                             viirs_geodata,
#                             0.12*ones(hls_ndims...),
#                             1e-2 * ones(hls_ndims...),
#                             model_pars_ff;
#                             nsamp = 50,
#                             window_buffer = 4,
#                             target_times = viirs_times, 
#                             spatial_mod = exp_cor,                                           
#                             obs_operator = unif_weighted_obs_operator_centroid,
#                             state_in_cov = false,
#                             cov_wt = 0.2,
#                             phi = 0.002,
#                             ar_par = 0.95,
#                             nb_coarse = 2.0,
#                             batchsize=1);

# @time fused_images_dbcs_pmap3, fused_sd_images_dbcs_pmap3 = coarse_fine_scene_fusion_pmap(hls_data,
#                             viirs_dbcs_data,
#                             hls_geodata, 
#                             viirs_geodata,
#                             window_geodata,
#                             0.12*ones(hls_ndims...),
#                             1e-2 * ones(hls_ndims...),
#                             model_pars_ff;
#                             nsamp = 20,
#                             window_buffer = 3,
#                             target_times = viirs_times, 
#                             spatial_mod = exp_cor,                                           
#                             obs_operator = unif_weighted_obs_operator_centroid,
#                             state_in_cov = false,
#                             cov_wt = 0.2,
#                             phi = 0.002,
#                             ar_par = 1.0,
#                             nb_coarse = 1.0,
#                             batchsize=1);

@time fused_images_cbias_pmap, fused_sd_images_cbias_pmap, fused_bias_images_cbias_pmap, fused_bias_sd_images_cbias_pmap = coarse_fine_scene_fusion_cbias_pmap(hls_data,
                            viirs_cbias_data,
                            hls_geodata, 
                            viirs_geodata,
                            0.3*ones(hls_ndims...),
                            1e-2 * ones(hls_ndims...),
                            zeros(viirs_ndims...),
                            1e-6 * ones(viirs_ndims...),
                            model_pars_ff;
                            nsamp = 100,
                            window_buffer = 7,
                            target_times = viirs_times, 
                            spatial_mod = exp_cor,                                           
                            obs_operator = unif_weighted_obs_operator_centroid,
                            state_in_cov = false,
                            cov_wt = 0.2,
                            phi = 0.002,
                            ar_par = 1.0,
                            nb_coarse = 2.0,
                            batchsize=1);

rmprocs(workers())

### old workflow
cov_raster_full = Raster(fill(NaN, nr, nc, 4), dims = (coarse_images.dims[1:2]...,Band(1:4)),missingval=fine_images.missingval)
cov_raster_full[:,:,1] = sp_var
cov_raster_full[:,:,2] .= viirs_csize[1]
cov_raster_full[:,:,3] .= 1e-10
cov_raster_full[:,:,4] .= 0.5;

@time fusion_results = coarse_fine_data_fusion(
                            coarse_images, 
                            fine_images, 
                            cov_raster_full,
                            prior = nothing,
                            target_times = viirs_times,
                            buffer_distance = 500.,
                            offset_ar = [1, 0.0], 
                            offset_var = [1e-6, 1e-6],    
                            default_mean = 0.12,
                            smooth = false
                        );


k=1
plot(Raster(fused_images_dbcs_pmap[:,:,k], dims=fusion_results.mean.dims[1:2]), layout=(1,2), title="new flow", size=(1000,400))
plot!(fusion_results.mean[:,:,k], subplot=2, title="old flow")

plot(Raster(fused_images_dbcs_pmap[:,:,k] .- fused_images_dbcs_pmap[:,:,k+1], dims=fusion_results.mean.dims[1:2]), clim=(-0.3,0.3), cmap=:bwr, layout=(1,3), title="new flow", size=(1500,400))
plot!(fusion_results.mean[:,:,k] .- fusion_results.mean[:,:,k+1], clim=(-0.3,0.3), cmap=:bwr, subplot=2, title="old flow")
plot!(Raster(fused_images_cbias_pmap[:,:,k] .- fused_images_cbias_pmap[:,:,k+1], dims=fusion_results.mean.dims[1:2]), clim=(-0.3,0.3), cmap=:bwr, subplot=3, title="new w/bias flow")


i=26
j=30

i=10
j=10
plot(viirs_times, fused_images_pmap[i,j,:][:], ribbon=2*fused_sd_images_pmap[i,j,:][:], markersize=2, label="new");
scatter!(viirs_times, coarse_images[Int(ceil(i/7)),Int(ceil(j/7)),:][:], markersize=2);
scatter!(viirs_times, fine_images[i,j,:][:], markersize=2)

plot(viirs_times, fused_images_bc_pmap[i,j,:][:], ribbon=2*fused_sd_images_bc_pmap[i,j,:][:], markersize=2, label="new");
scatter!(viirs_times, coarse_images[Int(ceil(i/7)),Int(ceil(j/7)),:][:], markersize=2);
scatter!(viirs_times, fine_images[i,j,:][:], markersize=2)

plot(viirs_times, fused_images_bcs_pmap[i,j,:][:], ribbon=2*fused_sd_images_bcs_pmap[i,j,:][:], markersize=2, label="new");
scatter!(viirs_times, coarse_images[Int(ceil(i/7)),Int(ceil(j/7)),:][:], markersize=2);
scatter!(viirs_times, fine_images[i,j,:][:], markersize=2)

plot(viirs_times, fusion_results.mean[i,j,:][:], ribbon=2*fusion_results.SD[i,j,:][:], markersize=2, label="old", clim=(0.2,0.4))
plot!(viirs_times, fused_images_cbias_pmap[i,j,:][:], ribbon=2*fused_sd_images_cbias_pmap[i,j,:][:], markersize=2, label="new w/bias", clim=(0.2,0.4))
plot!(viirs_times, fused_images_dbcs_pmap[i,j,:][:], ribbon=2*fused_sd_images_dbcs_pmap[i,j,:][:], markersize=2, label="new", clim=(0.2,0.4))
scatter!(viirs_times, viirs_array[Int(ceil(i/7)),Int(ceil(j/7)),:][:], markersize=2, clim=(0.2,0.4), label="viirs")
scatter!(viirs_times, viirs_dbcs_data.data[Int(ceil(i/7)),Int(ceil(j/7)),:][:], markersize=2, clim=(0.2,0.4), label="bias corrected viirs")
scatter!(viirs_times, fine_images[i,j,:][:], markersize=2, clim=(0.2,0.4), label = "hls")

plot(viirs_times, fusion_results.mean[i,j,:][:], ribbon=2*fusion_results.SD[i,j,:][:], markersize=2, label="old", clim=(0.2,0.4))
scatter!(viirs_times, coarse_images[Int(ceil(i/7)),Int(ceil(j/7)),:][:], markersize=2, clim=(0.2,0.4))
scatter!(viirs_times, fine_images[i,j,:][:], markersize=2, clim=(0.2,0.4))

plot(viirs_times, fusion_results.mean_bias[Int(ceil(i/7)),Int(ceil(j/7)),:][:], ribbon=2*fusion_results.SD_bias[i,j,:][:], markersize=2, label="new")
plot!(viirs_times, bias_field[Int(ceil(i/7)),Int(ceil(j/7)),:][:], label="new")

plot(viirs_times, fused_images_dbcs_pmap[i,j,:][:], ribbon=2*fused_sd_images_dbcs_pmap[i,j,:][:], markersize=2, label="phi=1.0", clim=(0.2,0.4))
plot!(viirs_times, fused_images_dbcs_pmap2[i,j,:][:], ribbon=2*fused_sd_images_dbcs_pmap2[i,j,:][:], markersize=2, label="phi=0.95", clim=(0.2,0.4))
plot!(viirs_times, fused_images_dbcs_pmap3[i,j,:][:], ribbon=2*fused_sd_images_dbcs_pmap3[i,j,:][:], markersize=2, label="phi=0.7", clim=(0.2,0.4))
plot!(viirs_times, fusion_results.mean[i,j,:][:], markersize=2, label="phi=0.7", clim=(0.2,0.4))

