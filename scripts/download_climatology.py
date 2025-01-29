import copernicusmarine
import dask
from dask.diagnostics import ProgressBar
import os

# Load xarray dataset
full_ds = copernicusmarine.open_dataset(
    dataset_id = 'cmems_mod_glo_phy-mnstd_my_0.25deg_P1M-m'
)

ds = full_ds.sel(latitude=slice(60, 90))
ds = ds.sel(depth=slice(0,500))
ds = ds.sel(time=slice('2003-01-01', '2024'))
ds = ds[['so_mean', 'thetao_mean', 'siconc_mean', 'sithick_mean', 'uo_mean', 'vo_mean', 'zos_mean']]
ds = ds.rename_vars({
    'so_mean': 'S', 
    'thetao_mean': 'T', 
    'siconc_mean': 'SIC', 
    'sithick_mean': 'SIT', 
    'uo_mean': 'U', 
    'vo_mean': 'V', 
    'zos_mean': 'SSH'
})
print('Selected dataset:')
print(ds)
ds_clim = ds.groupby('time.month').mean()
print()


with ProgressBar():
    ds_clim = ds_clim.compute()

dir = './data'
if not os.path.exists(dir):
    os.makedirs(dir)

if os.path.exists(dir + '/climatology.nc'):
    os.remove(dir + '/climatology.nc')

with ProgressBar():
    ds_clim.to_netcdf(dir + '/climatology.nc')