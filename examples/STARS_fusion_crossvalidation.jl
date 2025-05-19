using Dates
using Rasters
using STARSDataFusion
# using STARSDataFusion.STARSDataFusion
using LinearAlgebra
using Statistics
using Plots
using Distributed

addprocs(8)
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

scf = 4
nwindows = Int.(ceil.(hls_ndims./scf))
target_ndims = nwindows.*scf
window_geodata = STARSInstrumentGeoData(hls_origin .+ (scf-1.0)/2.0*hls_csize, scf*hls_csize, nwindows, 0, viirs_times)

### additive bias correction for viirs
fine_times = findall(mean(isnan.(fine_images), dims=(1,2))[:] .< 0.3)
fine_images_rs = resample(fine_images, to=coarse_images[:,:,1], method=:average)

### other way to resample
fine_images_fs = disaggregate(fine_images, (5,5,1))
fine_images_fs_rs = resample(fine_images_fs, to=coarse_images[:,:,1], method=:average)

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

viirs_cbias_data = STARSInstrumentData(viirs_array, 0.0, 1e-6, true, [1.0,1e-6], abs.(viirs_csize), viirs_times, [1. 1.])

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

### for old workflow
cov_raster_full = Raster(fill(NaN, nr, nc, 4), dims = (coarse_images.dims[1:2]...,Band(1:4)),missingval=fine_images.missingval)
cov_raster_full[:,:,1] = sp_var
cov_raster_full[:,:,2] .= viirs_csize[1]
cov_raster_full[:,:,3] .= 1e-10
cov_raster_full[:,:,4] .= 0.5;

### cross-validation leave-out-HLS predictions
fused_nb_cv = zeros((size(hls_array)[1:2]...,length(fine_times)-1))
fused_bc_cv = similar(fused_nb_cv)
fused_bcs_cv = similar(fused_nb_cv)
fused_dbcs_cv = similar(fused_nb_cv)
fused_cbias_cv = similar(fused_nb_cv)
fused_old_cv = similar(fused_nb_cv)

for (i,j) in enumerate(fine_times[2:end])
    hls_data = STARSInstrumentData(hls_array[:,:,1:end .≠ (i+1)], 0.0, 1e-6, false, nothing, abs.(hls_csize), hls_times[1:end .≠ (i+1)], [1. 1.])
    hls_geodata = STARSInstrumentGeoData(hls_origin, hls_csize, hls_ndims, 0, hls_times[1:end .≠ (i+1)])
    bias_field2 = bias_field[:,:,:]
    bias_field2[:,:,j:size(bias_field2,3)] .= bias_field[:,:,j-1]
    for k in (j+1):length(viirs_times)
        kp = findall(.!isnan.(errs[:,:,k]))
        if length(kp) > 0
            bias_field2[kp,k:size(bias_field2,3)] .= errs[kp,k]
        end
    end
    # bias_field3 = Array(fusion_results.mean_bias)
    viirs_dbcs_data = STARSInstrumentData(viirs_array .- bias_field2, 0.0, 1e-6, false, nothing, abs.(viirs_csize), viirs_times, [1. 1.])

    fused_images_pmap, fused_sd_images_pmap = coarse_fine_scene_fusion_pmap(hls_data,
                                viirs_data,
                                hls_geodata, 
                                viirs_geodata,
                                viirs_geodata,
                                0.12*ones(hls_ndims...),
                                1e-2 * ones(hls_ndims...),
                                model_pars_ff;
                                nsamp = 20,
                                window_buffer = 4,
                                target_times = 1:j, 
                                spatial_mod = exp_cor,                                           
                                obs_operator = unif_weighted_obs_operator_centroid,
                                state_in_cov = false,
                                cov_wt = 0.2,
                                phi = 0.002,
                                nb_coarse = 2.0,
                                batchsize=1);

    fused_images_bc_pmap, fused_sd_images_bc_pmap = coarse_fine_scene_fusion_pmap(hls_data,
                                viirs_bc_data,
                                hls_geodata, 
                                viirs_geodata,
                                viirs_geodata,
                                0.12*ones(hls_ndims...),
                                1e-2 * ones(hls_ndims...),
                                model_pars_ff;
                                nsamp = 20,
                                window_buffer = 4,
                                target_times = 1:j,
                                spatial_mod = exp_cor,                                           
                                obs_operator = unif_weighted_obs_operator_centroid,
                                state_in_cov = false,
                                cov_wt = 0.2,
                                phi = 0.002,
                                nb_coarse = 2.0,
                                batchsize=1);

    fused_images_bcs_pmap, fused_sd_images_bcs_pmap = coarse_fine_scene_fusion_pmap(hls_data,
                                viirs_bcs_data,
                                hls_geodata, 
                                viirs_geodata,
                                viirs_geodata,
                                0.12*ones(hls_ndims...),
                                1e-2 * ones(hls_ndims...),
                                model_pars_ff;
                                nsamp = 20,
                                window_buffer = 4,
                                target_times = 1:j,
                                spatial_mod = exp_cor,                                           
                                obs_operator = unif_weighted_obs_operator_centroid,
                                state_in_cov = false,
                                cov_wt = 0.2,
                                phi = 0.002,
                                nb_coarse = 2.0,
                                batchsize=1);

    fused_images_dbcs_pmap, fused_sd_images_dbcs_pmap = coarse_fine_scene_fusion_pmap(hls_data,
                                viirs_dbcs_data,
                                hls_geodata, 
                                viirs_geodata,
                                viirs_geodata,
                                0.12*ones(hls_ndims...),
                                1e-2 * ones(hls_ndims...),
                                model_pars_ff;
                                nsamp = 20,
                                window_buffer = 4,
                                target_times = 1:j,
                                spatial_mod = exp_cor,                                           
                                obs_operator = unif_weighted_obs_operator_centroid,
                                state_in_cov = false,
                                cov_wt = 0.2,
                                phi = 0.002,
                                nb_coarse = 2.0,
                                batchsize=1);

    fused_images_cbias_pmap, fused_sd_images_cbias_pmap, fused_bias_images_cbias_pmap, fused_bias_sd_images_cbias_pmap = coarse_fine_scene_fusion_cbias_pmap(hls_data,
                            viirs_cbias_data,
                            hls_geodata, 
                            viirs_geodata,
                            0.12*ones(hls_ndims...),
                            1e-2 * ones(hls_ndims...),
                            zeros(viirs_ndims...),
                            1e-6 * ones(viirs_ndims...),
                            model_pars_ff;
                            nsamp = 20,
                            window_buffer = 3,
                            target_times = 1:j, 
                            spatial_mod = exp_cor,                                           
                            obs_operator = unif_weighted_obs_operator_centroid,
                            state_in_cov = false,
                            cov_wt = 0.2,
                            phi = 0.002,
                            ar_par = 1.0,
                            nb_coarse = 2.0,
                            batchsize=1);

    fine_images2 = fine_images[:,:,:]
    fine_images2[:,:,j] .= NaN
    fusion_results = coarse_fine_data_fusion(
                                coarse_images, 
                                fine_images2, 
                                cov_raster_full,
                                prior = nothing,
                                target_times = 1:j,
                                buffer_distance = 100.,
                                offset_ar = [1, 0.0], 
                                offset_var = [1e-6, 1e-6],    
                                default_mean = 0.12,
                                smooth = false
                            );


    fused_nb_cv[:,:,i] = fused_images_pmap[:,:,j]
    fused_bc_cv[:,:,i] = fused_images_bc_pmap[:,:,j]
    fused_bcs_cv[:,:,i] = fused_images_bcs_pmap[:,:,j]
    fused_dbcs_cv[:,:,i] = fused_images_dbcs_pmap[:,:,j]
    fused_cbias_cv[:,:,i] = fused_images_cbias_pmap[:,:,j]
    fused_old_cv[:,:,i] = Array(fusion_results.mean[:,:,j])
end

rmprocs(workers())

errs_nb = fused_nb_cv .- hls_array[:,:,2:end]
errs_bc = fused_bc_cv .- hls_array[:,:,2:end]
errs_bcs = fused_bcs_cv .- hls_array[:,:,2:end]
errs_dbcs = fused_dbcs_cv .- hls_array[:,:,2:end]
errs_cbias = fused_cbias_cv .- hls_array[:,:,2:end]
errs_old = fused_old_cv .- hls_array[:,:,2:end]
errs_hls = hls_array[:,:,1:end-1] .- hls_array[:,:,2:end]

rmse_nb = sqrt.(mapslices(nanmean, errs_nb.^2, dims=(1,2))[:])
rmse_bc = sqrt.(mapslices(nanmean, errs_bc.^2, dims=(1,2))[:])
rmse_bcs = sqrt.(mapslices(nanmean, errs_bcs.^2, dims=(1,2))[:])
rmse_dbcs = sqrt.(mapslices(nanmean, errs_dbcs.^2, dims=(1,2))[:])
rmse_cbias = sqrt.(mapslices(nanmean, errs_cbias.^2, dims=(1,2))[:])

rmse_old = sqrt.(mapslices(nanmean, errs_old.^2, dims=(1,2))[:])
rmse_lhls = sqrt.(mapslices(nanmean, errs_hls.^2, dims=(1,2))[:])


rmse_map_nb = sqrt.(mapslices(nanmean, errs_nb.^2, dims=3))
rmse_map_bc = sqrt.(mapslices(nanmean, errs_bc.^2, dims=3))
rmse_map_bcs = sqrt.(mapslices(nanmean, errs_bcs.^2, dims=3))
rmse_map_dbcs = sqrt.(mapslices(nanmean, errs_dbcs.^2, dims=3))
rmse_map_old = sqrt.(mapslices(nanmean, errs_old.^2, dims=3))
rmse_map_lhls = sqrt.(mapslices(nanmean, errs_hls.^2, dims=3))

bias_nb = mapslices(nanmean, errs_nb, dims=(1,2))[:]
bias_bc = mapslices(nanmean, errs_bc.^2, dims=(1,2))[:]
bias_bcs = mapslices(nanmean, errs_bcs.^2, dims=(1,2))[:]
bias_dbcs = mapslices(nanmean, errs_dbcs.^2, dims=(1,2))[:]
bias_old = mapslices(nanmean, errs_old.^2, dims=(1,2))[:]
bias_lhls = mapslices(nanmean, errs_hls.^2, dims=(1,2))[:]


# scatter(fine_times[2:end], rmse_nb, label="no bias", markersize=3)
scatter(fine_times[2:end], rmse_bc, label="constant bias", markersize=3)
# scatter!(fine_times[2:end], rmse_bcs, label="constant pixel bias", markersize=3)
scatter!(fine_times[2:end], rmse_dbcs, label="dynamic pixel bias", markersize=3)
scatter!(fine_times[2:end], rmse_cbias, label="dynamic pixel bias", markersize=3)
# scatter!(fine_times[2:end], rmse_old, label="old", markersize=3)
scatter!(fine_times[2:end], rmse_lhls, label="last hls", markersize=3)


scatter(fine_times[2:end], bias_nb, label="no bias", markersize=3)
scatter!(fine_times[2:end], bias_bc, label="constant bias", markersize=3)
scatter!(fine_times[2:end], bias_bcs, label="constant pixel bias", markersize=3)
scatter!(fine_times[2:end], bias_dbcs, label="dynamic pixel bias", markersize=3)
scatter!(fine_times[2:end], bias_old, label="old", markersize=3)
scatter!(fine_times[2:end], bias_lhls, label="last hls", markersize=3)

scatter(fine_times[2:end], bias_bc, label="constant bias", markersize=3)
scatter!(fine_times[2:end], bias_bcs, label="constant pixel bias", markersize=3)
scatter!(fine_times[2:end], bias_dbcs, label="dynamic pixel bias", markersize=3)
scatter!(fine_times[2:end], bias_old, label="old", markersize=3)
scatter!(fine_times[2:end], bias_lhls, label="last hls", markersize=3)

scatter(fine_times[2:end], rmse_nb .- rmse_old);Plots.hline!([0])
scatter(fine_times[2:end], rmse_bc .- rmse_old);Plots.hline!([0])
scatter(fine_times[2:end], rmse_bcs .- rmse_old);Plots.hline!([0])
scatter(fine_times[2:end], rmse_dbcs .- rmse_old);Plots.hline!([0])

scatter(fine_times[2:end], rmse_nb .- rmse_lhls, ylim=(-0.02,0.02));Plots.hline!([0])
scatter(fine_times[2:end], rmse_bc .- rmse_lhls, ylim=(-0.02,0.02));Plots.hline!([0])
scatter(fine_times[2:end], rmse_bcs .- rmse_lhls, ylim=(-0.02,0.02));Plots.hline!([0])
scatter(fine_times[2:end], rmse_dbcs .- rmse_lhls, ylim=(-0.02,0.02));Plots.hline!([0])
scatter(fine_times[2:end], rmse_old .- rmse_lhls, ylim=(-0.02,0.02));Plots.hline!([0])

heatmap(rmse_map_nb[:,:,1], clim=(0,0.15))
heatmap(rmse_map_bc[:,:,1], clim=(0,0.15))
heatmap(rmse_map_bcs[:,:,1], clim=(0,0.15))
heatmap(rmse_map_dbcs[:,:,1], clim=(0,0.15))
heatmap(rmse_map_old[:,:,1], clim=(0,0.15))
heatmap(rmse_map_lhls[:,:,1], clim=(0,0.15))

histogram(rmse_map_nb[:], xlim=(0,0.15))
histogram(rmse_map_bc[:], xlim=(0,0.15))
histogram(rmse_map_bcs[:], xlim=(0,0.15))
histogram(rmse_map_dbcs[:], xlim=(0,0.15))
histogram(rmse_map_old[:], xlim=(0,0.15))
histogram(rmse_map_lhls[:], xlim=(0,0.15))

k=17
heatmap(errs_hls[:,:,k], cmap=:bwr, clim = (-0.2,0.2), layout=(2,3), size=(1400,600), title="last hls")
heatmap!(errs_nb[:,:,k], cmap=:bwr, clim = (-0.2,0.2), subplot=2, title="no bias")
heatmap!(errs_bc[:,:,k], cmap=:bwr, clim = (-0.2,0.2), subplot=3, title="constant")
heatmap!(errs_bcs[:,:,k], cmap=:bwr, clim = (-0.2,0.2), subplot=4, title="scene")
heatmap!(errs_dbcs[:,:,k], cmap=:bwr, clim = (-0.2,0.2), subplot=5, title="dynamic")
heatmap!(errs_old[:,:,k], cmap=:bwr, clim = (-0.2,0.2), subplot=6, title="old")

scatter(fine_times[2:end], rmse_nb, label="no bias");
scatter!(fine_times[2:end], rmse_lhls, label="last HLS")

scatter(fine_times[2:end], rmse_bc, label="constant bias");
scatter!(fine_times[2:end], rmse_lhls, label="last HLS")

scatter(fine_times[2:end], rmse_dbcs, label="dynamic bias");
scatter!(fine_times[2:end], rmse_old, label="old")

scatter(fine_times[2:end], rmse_bcs, label="scene bias");
scatter!(fine_times[2:end], rmse_lhls, label="last HLS")

scatter(fine_times[2:end], rmse_old, label="old bias");
scatter!(fine_times[2:end], rmse_lhls, label="last HLS")

#### what's going on
ii=26
jj=30
plot(viirs_times[1:j], fused_images_dbcs_pmap[ii,jj,:][:], ribbon=2*fused_sd_images_dbcs_pmap[ii,jj,:][:], markersize=2, label="new", clim=(0.2,0.4), size=(1000,600))
plot!(viirs_times[1:j], fused_images_bc_pmap[ii,jj,:][:], ribbon=2*fused_sd_images_bc_pmap[ii,jj,:][:], markersize=2, label="new", clim=(0.2,0.4), size=(1000,600))
plot!(viirs_times[1:j], fusion_results.mean[ii,jj,:][:], ribbon=2*fusion_results.SD[ii,jj,:][:], markersize=2, label="old", clim=(0.2,0.4))
scatter!(viirs_times[1:j], viirs_array[Int(ceil(ii/7)),Int(ceil(jj/7)),1:j][:], markersize=2, clim=(0.2,0.4), label="viirs")
scatter!(viirs_times[1:j], viirs_dbcs_data.data[Int(ceil(ii/7)),Int(ceil(jj/7)),1:j][:], markersize=2, clim=(0.2,0.4), label="bias corrected viirs")
scatter!(viirs_times[1:j], fine_images[ii,jj,:][:], markersize=2, clim=(0.2,0.4), label = "hls")
scatter!(viirs_times[j:j],[fused_dbcs_cv[ii,jj,i]])

scatter(Array(fusion_results.mean_bias)[Int(ceil(ii/7)),Int(ceil(jj/7)),:], label="old");
scatter!(bias_field[Int(ceil(ii/7)),Int(ceil(jj/7)),:], label="new")

l=12
scatter(bias_field2[:,:,l][:], fusion_results.mean_bias[:,:,l][:], markersize=1);Plots.abline!(1,0)


plot(viirs_times[1:j], fused_images_dbcs_pmap[ii,jj,:][:], markersize=2, label="new", clim=(0.2,0.4), size=(1000,600))
# scatter!(viirs_times[1:j], viirs_dbcs_data.data[Int(ceil(ii/7)),Int(ceil(jj/7)),1:j][:], markersize=2, clim=(0.2,0.4), label="bias corrected viirs")
scatter!(viirs_times[1:j], viirs_array[Int(ceil(ii/7)),Int(ceil(jj/7)),1:j][:], markersize=2, clim=(0.2,0.4), label="viirs")
scatter!(viirs_times[1:j], bias_field2[Int(ceil(ii/7)),Int(ceil(jj/7)),1:j][:], markersize=2, label="bias")
scatter!(viirs_times[1:j], fine_images[ii,jj,:][:], markersize=2, clim=(0.2,0.4), label = "hls")

plot(viirs_times[1:j], fusion_results.mean[ii,jj,:][:], markersize=2, label="new", clim=(0.2,0.4), size=(1000,600))
scatter!(viirs_times[1:j], viirs_array[Int(ceil(ii/7)),Int(ceil(jj/7)),1:j][:], markersize=2, clim=(0.2,0.4), label="viirs")
scatter!(viirs_times[1:j], Array(fusion_results.mean_bias)[Int(ceil(ii/7)),Int(ceil(jj/7)),1:j][:], markersize=2, label="bias")
scatter!(viirs_times[1:j], fine_images[ii,jj,:][:], markersize=2, clim=(0.2,0.4), label = "hls")


heatmap(fused_images_dbcs_pmap[:,:,3] .- fused_images_dbcs_pmap[:,:,2], cmap=:bwr, clim=(-0.1,0.1))