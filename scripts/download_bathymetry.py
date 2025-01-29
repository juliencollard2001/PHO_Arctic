import copernicusmarine
import dask
from dask.diagnostics import ProgressBar
import os

full_ds = copernicusmarine.open_dataset(
    dataset_id = 'cmems_mod_glo_phy_anfc_0.083deg_static'
)
bathy = full_ds['deptho'].sel(latitude=slice(60,91))
bathy_compressed = bathy.coarsen(latitude=3, longitude=3, boundary='pad').mean()


with ProgressBar():
    bathy_compressed = bathy_compressed.compute()

dir = './data'
if not os.path.exists(dir):
    os.makedirs(dir)

if os.path.exists(dir + '/bathy.nc'):
    os.remove(dir + '/bathy.nc')

with ProgressBar():
    bathy_compressed.to_netcdf(dir + '/bathy.nc')