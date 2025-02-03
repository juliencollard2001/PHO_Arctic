import xarray as xr # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as colors # type: ignore
import cmocean.cm as cmo # type: ignore
import cartopy.crs as ccrs # type: ignore
import cartopy.feature as cfeature # type: ignore
import cartopy.io.shapereader as shpreader # type: ignore
from tqdm import tqdm # type: ignore

def load_climatology_with_deptho():
    ds = xr.open_dataset('data/climatology.nc').load()
    bathy_ds = xr.open_dataset('data/bathy.nc').load()
    bathy_ds['latitude'] = ds.latitude
    bathy_ds['longitude'] = ds.longitude
    ds = xr.merge([ds, bathy_ds])
    return ds

def load_surface_data():
    ds = xr.open_dataset('data/surface.nc').load()
    return ds

def spherical_to_cartesian(lat, lon, rad=False):
    if not rad:
        lat = np.radians(lat)
        lon = np.radians(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=0)

def spherical_distance(lat1, lon1, lat2, lon2, rad=False):
    if not rad:
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
    p1 = spherical_to_cartesian(lat1, lon1, rad)
    p2 = spherical_to_cartesian(lat2, lon2, rad)
    return np.arccos(np.dot(p1, p2))


def cross_section(
        ds: xr.Dataset, 
        lon1: float, 
        lat1: float, 
        lon2: float, 
        lat2: float, 
        N : int = 50, 
        plot: bool = True
    ) -> xr.Dataset:
    
    P_start = spherical_to_cartesian(lat1, lon1, rad=False)
    P_end = spherical_to_cartesian(lat2, lon2, rad=False)
    delta = np.arccos(np.clip(np.dot(P_start, P_end), -1.0, 1.0))

    # Finding all (lat, lon) pairs within on the start-end line
    t = np.linspace(0, 1, N)
    P_t = (np.sin((1 - t)[None,:] * delta) * P_start[:,None] + np.sin(t * delta)[None,:] * P_end[:,None]) / np.sin(delta)
    x, y, z = P_t
    line_lats = np.degrees(np.arcsin(z))
    line_lons = np.degrees(np.arctan2(y, x))

    lats, lons = np.meshgrid(ds.latitude, ds.longitude)
    lats, lons = lats.ravel(), lons.ravel()

    xs = np.cos(np.radians(lats)) * np.cos(np.radians(lons))
    ys = np.cos(np.radians(lats)) * np.sin(np.radians(lons))
    zs = np.sin(np.radians(lats))

    Ps = np.stack([xs, ys, zs], axis=0)
    D = np.arccos(np.clip(np.dot(Ps.T, P_t), -1.0, 1.0))

    if plot:
        # Showing cross section on a map
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()}, figsize=(5, 5))
        ax.coastlines()
        ax.gridlines()
        ax.add_feature(cfeature.LAND)
        ax.set_extent([-180, 180, 55, 90], ccrs.PlateCarree())
        ax.pcolormesh(ds.longitude, ds.latitude, ds['deptho'].values, transform=ccrs.PlateCarree(), cmap=cmo.deep)
        points = ax.scatter(line_lons, line_lats, c=np.arange(0,N), cmap='gist_rainbow', transform=ccrs.PlateCarree())
        plt.colorbar(points, ax=ax, orientation='vertical', label='Cross section indice', shrink=0.5)
        plt.title(f'Cross section from ({lon1}, {lat1}) to ({lon2}, {lat2})')
        plt.show()

    cs_dss = []

    for t in tqdm(range(N)):
        sub_D = D[:,t]
        idxs = sub_D.argsort()
        best_idx = idxs[0]
        best_dist = sub_D[best_idx]
        second_best_idx = idxs[1]
        second_best_dist = sub_D[second_best_idx]

        w = second_best_dist / (second_best_dist + best_dist)

        best_lat = lats[best_idx]
        best_lon = lons[best_idx]
        second_best_lat = lats[second_best_idx]
        second_best_lon = lons[second_best_idx]

        cs_ds = w * ds.sel(latitude=best_lat, longitude=best_lon, method='nearest') + (1-w) * ds.sel(latitude=second_best_lat, longitude=second_best_lon,method='nearest') 
        cs_dss.append(cs_ds)

    cs_ds = xr.concat(cs_dss, dim='cross_section_idx', coords='minimal', compat='override')
    cs_ds['dist'] = np.arange(N) * delta * np.pi * 6371 / N
    cs_ds['latitude'] = line_lats
    cs_ds['longitude'] = line_lons

    return cs_ds
