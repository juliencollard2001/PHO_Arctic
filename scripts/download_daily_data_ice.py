import copernicusmarine
import dask
from dask.diagnostics import ProgressBar
import os

# Load xarray dataset
full_ds = copernicusmarine.open_dataset(
    dataset_id = 'cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m'
)

ds = full_ds.sel(latitude=slice(84, 86))
ds = ds.sel(longitude=slice(-1, 1))
ds = ds.sel(time=slice('2003-01-01', '2024'))
ds = ds[['sithick_mean']]
ds = ds.rename_vars({'sithick_mean': 'SIT'})
print('Selected dataset:')
print(ds)
print(f'ds size in GB: {ds.nbytes / 1e9}')
print()


dir = './data'
if not os.path.exists(dir):
    os.makedirs(dir)

if os.path.exists(dir + '/surface.nc'):
    os.remove(dir + '/surface.nc')

with ProgressBar():
    ds.to_netcdf(dir + '/surface.nc')