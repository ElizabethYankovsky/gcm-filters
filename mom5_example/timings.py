## Laplacian filtering on MOM5 vorticity


import os
import sys
import time

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from read_data import read_data
from scipy.ndimage import gaussian_filter
from vorticity import compute_vorticity

from gcm_filters import filter
from gcm_filters.kernels import GridType


lats = slice(
    -20.0, 70.0
)  # interested in 0 to 60 but expanding to alleviate boundary effects
lons = slice(
    -260, -170
)  # interested in -240:-180 but expanding to alleviate boundary effects
latsT = slice(-20.0, 70.0)
lonsT = slice(-260.0, -170.0)


n_runs = 100
sigmas = [16]
cut_offs = [2, 4, 8, 16, 32]
n_steps = [4, 8, 16, 32, 64]


# Arrays for storing compute times and "errors" (not used)
results_lapl = np.zeros((len(sigmas), len(n_steps), n_runs))
results_lapl2 = np.zeros((len(sigmas), len(n_steps), n_runs))
results_scipy = np.zeros((len(sigmas), len(cut_offs), n_runs))

error_lapl = np.zeros((len(sigmas), len(n_steps), n_runs))
error_lapl2 = np.zeros((len(sigmas), len(n_steps), n_runs))
error_scipy = np.zeros((len(sigmas), len(cut_offs), n_runs))

# Arrays for storing compute times when using the default n_step
results_lapl_d = np.zeros((len(sigmas), n_runs))
error_lapl_d = np.zeros((len(sigmas), n_runs))
results_lapl2_d = np.zeros((len(sigmas), n_runs))
error_lapl2_d = np.zeros((len(sigmas), n_runs))


data_location = "/scratch/ag7531/shareElizabeth/"
# data_location = "/media/arthur/DATA/Data sets/CM2.6"

grid_filename = "grid_dataforeli"
uv_filename = "uv_dataforeli"

grid_data, data = read_data(data_location, uv_filename, grid_filename)
grid_data = grid_data.compute().reset_coords()
data = data[["usurf", "vsurf"]].sel(xu_ocean=lons, yu_ocean=lats).isel(time=0)
data_saved = data
grid_data_save = grid_data

data = data.sel(yu_ocean=lats, xu_ocean=lons)
print("done")


grid_data = grid_data[["dxt", "dyt", "dxu", "dyu", "area_u", "wet"]]
# Here we need dxt and dyt to be on the velocity grid
velocity_coords = grid_data[["yu_ocean", "xu_ocean"]]
grid_data["dxt"] = xr.DataArray(
    data=grid_data["dxt"], dims=("yu_ocean", "xu_ocean"), coords=velocity_coords
)
grid_data["dyt"] = xr.DataArray(
    data=grid_data["dyt"], dims=("yu_ocean", "xu_ocean"), coords=velocity_coords
)
grid_data_wet = np.asarray(grid_data["wet"].values, dtype=np.float32)
grid_data["wet"] = xr.DataArray(
    data=grid_data_wet, dims=("yu_ocean", "xu_ocean"), coords=velocity_coords
)
grid_data = grid_data.sel(xu_ocean=lons, yu_ocean=lats)
del grid_data["xt_ocean"]
del grid_data["yt_ocean"]
print(grid_data)

data = data.compute()
grid_data = grid_data.compute()

data2 = data.fillna(0.0)


for i_sigma, sigma in enumerate(sigmas):
    truth = xr.apply_ufunc(
        lambda x: gaussian_filter(x, sigma=sigma / np.sqrt(12), truncate=32),
        data2["usurf"],
    )
    for i_nsteps, n_step in enumerate(n_steps):
        mom5_filterU = filter.Filter(
            sigma,
            dx_min=1,
            n_steps=n_step,
            filter_shape=filter.FilterShape.GAUSSIAN,
            grid_vars=dict(wet_mask=grid_data["wet"]),
            grid_type=GridType.CARTESIAN_WITH_LAND,
        )
        mom5_filterU2 = filter.Filter(
            sigma,
            dx_min=1,
            n_steps=n_step,
            filter_shape=filter.FilterShape.GAUSSIAN,
            grid_vars=dict(),
            grid_type=GridType.CARTESIAN,
        )
        for i_run in range(n_runs):
            t0 = time.process_time()
            res = mom5_filterU.apply(data["usurf"], ["yu_ocean", "xu_ocean"])
            t1 = time.process_time()
            results_lapl[i_sigma, i_nsteps, i_run] = t1 - t0
            error_lapl[i_sigma, i_nsteps, i_run] = np.mean(abs(res - truth))
            # For cartesian
            t0 = time.process_time()
            res = mom5_filterU2.apply(data["usurf"], ["yu_ocean", "xu_ocean"])
            t1 = time.process_time()
            results_lapl2[i_sigma, i_nsteps, i_run] = t1 - t0
            error_lapl2[i_sigma, i_nsteps, i_run] = np.mean(abs(res - truth))
for i_sigma, sigma in enumerate(sigmas):
    truth = xr.apply_ufunc(
        lambda x: gaussian_filter(x, sigma=sigma / np.sqrt(12), truncate=32),
        data2["usurf"],
    )
    for i_cutoff, cut_off in enumerate(cut_offs):
        for i_run in range(n_runs):
            t2 = time.process_time()
            res2 = xr.apply_ufunc(
                lambda x: gaussian_filter(
                    x, sigma=(sigma / np.sqrt(12)), truncate=cut_off
                ),
                data2["usurf"],
            )
            t3 = time.process_time()
            results_scipy[i_sigma, i_cutoff, i_run] = t3 - t2
            error_scipy[i_sigma, i_cutoff, i_run] = np.mean(abs(res2 - truth))

mom5_filterU = filter.Filter(
    sigma,
    dx_min=1,
    filter_shape=filter.FilterShape.GAUSSIAN,
    grid_vars=dict(wet_mask=grid_data["wet"]),
    grid_type=GridType.CARTESIAN_WITH_LAND,
)
mom5_filterU2 = filter.Filter(
    sigma,
    dx_min=1,
    filter_shape=filter.FilterShape.GAUSSIAN,
    grid_vars=dict(),
    grid_type=GridType.CARTESIAN,
)

for i_run in range(n_runs):
    t0 = time.process_time()
    res = mom5_filterU.apply(data["usurf"], ["yu_ocean", "xu_ocean"])
    t1 = time.process_time()
    results_lapl_d[i_sigma, i_run] = t1 - t0
    error_lapl_d[i_sigma, i_run] = np.mean(abs(res - truth))
    # cartesian
    t0 = time.process_time()
    res = mom5_filterU2.apply(data["usurf"], ["yu_ocean", "xu_ocean"])
    t1 = time.process_time()
    results_lapl2_d[i_sigma, i_run] = t1 - t0
    error_lapl2_d[i_sigma, i_run] = np.mean(abs(res - truth))


mean_lapl = np.mean(results_lapl[0, ...], axis=-1)
mean_lapl2 = np.mean(results_lapl2[0, ...], axis=-1)
mean_scipy = np.mean(results_scipy[0, ...], axis=-1)
mean_lapl_d = np.mean(results_lapl_d[0, ...], axis=-1)
mean_lapl2_d = np.mean(results_lapl2_d[0, ...], axis=-1)


std_lapl = np.std(results_lapl[0, ...], axis=-1)
std_lapl2 = np.std(results_lapl2[0, ...], axis=-1)
std_scipy = np.std(results_scipy[0, ...], axis=-1)
std_lapl_d = np.std(results_lapl_d[0, ...], axis=-1)


plt.plot(range(len(cut_offs)), mean_lapl)
plt.plot(range(len(cut_offs)), mean_lapl2)
plt.plot(range(len(cut_offs)), mean_scipy)

plt.legend(("CARTESIAN_WITH_LAND", "CARTESIAN", "SCIPY"))

plt.xticks(
    range(len(cut_offs)), ["\n\n".join(map(str, x)) for x in zip(n_steps, cut_offs)]
)


plt.fill_between(
    range(len(cut_offs)),
    mean_lapl - 1.96 * std_lapl,
    mean_lapl + 1.96 * std_lapl,
    alpha=0.3,
)

plt.fill_between(
    range(len(cut_offs)),
    mean_lapl2 - 1.96 * std_lapl2,
    mean_lapl2 + 1.96 * std_lapl2,
    alpha=0.3,
)

plt.fill_between(
    range(len(cut_offs)),
    mean_scipy - 1.96 * std_scipy,
    mean_scipy + 1.96 * std_scipy,
    alpha=0.3,
)

n_steps_d = (
    2 * mom5_filterU.filter_spec.n_bih_steps + mom5_filterU.filter_spec.n_lap_steps
)
n_steps_d = np.log(n_steps_d) / np.log(2) - np.log(n_steps[0]) / np.log(2)
plt.plot(n_steps_d, mean_lapl_d, "*", markersize=15)
plt.plot(n_steps_d, mean_lapl2_d, "*", markersize=15)


plt.ylabel("s")
plt.xlabel("Number of steps / Number of stds for truncation")

plt.show()
plt.savefig("/scratch/ag7531/figure_gcm_filters.jpg", dpi=400)
