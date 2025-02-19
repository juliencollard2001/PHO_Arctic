import xarray as xr # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as colors # type: ignore
import cmocean.cm as cmo # type: ignore
import cartopy.crs as ccrs # type: ignore
import cartopy.feature as cfeature # type: ignore
import cartopy.io.shapereader as shpreader # type: ignore
from tqdm import tqdm # type: ignore
import pandas as pd # type: ignore
import gsw

def load_climatology_with_deptho():
    ds = xr.open_dataset('data/climatology.nc').load()
    bathy_ds = xr.open_dataset('data/bathy.nc').load()
    bathy_ds['latitude'] = ds.latitude
    bathy_ds['longitude'] = ds.longitude
    ds = xr.merge([ds, bathy_ds])
    return ds

def load_surface_data():
    ds = xr.open_dataset('data/surface.nc')
    return ds

def load_atm_data():
    ds1 = xr.open_dataset('data/era5_1.nc')
    ds2 = xr.open_dataset('data/era5_2.nc')
    ds2['valid_time'] = ds1['valid_time']
    ds = xr.merge([ds1, ds2])
    ds = ds.rename({'valid_time': 'time'}).sortby('latitude').drop_vars(['number', 'expver'])
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
    dist = np.arange(N) * delta * np.pi * 6371 / N
    cs_ds['dist'] = dist
    cs_ds['latitude'] = line_lats
    cs_ds['longitude'] = line_lons

    derivee_lon = np.zeros(N)  #Dérivée selon un schéma centré
    derivee_lat = np.zeros(N)  #Dérivée selon un schéma centré
    for i in range(N) :
        if i == 0 :
            derivee_lat[i] = (line_lats[i + 1] - line_lats[i] ) / (dist[i + 1] - dist[i])
            derivee_lon[i] = (line_lons[i + 1] - line_lons[i] ) / (dist[i + 1] - dist[i])
        elif i == N-1 :
            derivee_lat[i] = (line_lats[i] - line_lats[i - 1] ) / (dist[i] - dist[i - 1])
            derivee_lon[i] = (line_lons[i] - line_lons[i - 1] ) / (dist[i] - dist[i - 1])
        else :
            derivee_lat[i] = (line_lats[i + 1] - line_lats[i - 1] ) / (dist[i + 1] - dist[i - 1])
            derivee_lon[i] = (line_lons[i + 1] - line_lons[i - 1] ) / (dist[i + 1] - dist[i - 1])

    normalization = np.sqrt(derivee_lon**2 + derivee_lat**2) #Facteur de normalisation

    cs_ds['normal_meridional'] = ('cross_section_idx', derivee_lon / normalization)
    cs_ds['normal_zonal'] = ('cross_section_idx', - derivee_lat / normalization)


    return cs_ds

def vecteurs_coupe(
       cs_ds : xr.Dataset,
    ) -> xr.Dataset :

    # PARTIE DÉTERMINATION VECTEURS UNITAIRES POUR DECRIRE LA SECTION
        
    dist = cs_ds['dist']
    lat = cs_ds['latitude']
    lon = cs_ds['longitude']

    N = len(dist)
    derivee_lon = np.zeros(N)  #Dérivée selon un schéma centré
    derivee_lat = np.zeros(N)  #Dérivée selon un schéma centré
    for i in range(N) :
        if i == 0 :
            derivee_lat[i] = (lat[i + 1] - lat[i] ) / (dist[i + 1] - dist[i])
            derivee_lon[i] = (lon[i + 1] - lon[i] ) / (dist[i + 1] - dist[i])
        elif i == N-1 :
            derivee_lat[i] = (lat[i] - lat[i - 1] ) / (dist[i] - dist[i - 1])
            derivee_lon[i] = (lon[i] - lon[i - 1] ) / (dist[i] - dist[i - 1])

        else :
            derivee_lat[i] = (lat[i + 1] - lat[i - 1] ) / (dist[i + 1] - dist[i - 1])
            derivee_lon[i] = (lon[i + 1] - lon[i - 1] ) / (dist[i + 1] - dist[i - 1])

    normalization = np.sqrt(derivee_lon**2 + derivee_lat**2) #Facteur de normalisation

    vec_t = [derivee_lon, derivee_lat]/normalization
    vec_n = [-1*derivee_lat, derivee_lon]/normalization

    # PARTIE CALCUL VITESSES

    u = cs_ds.U
    v = cs_ds.V

    #vitesse_t = np.zeros_like(u)
    vitesse_n = np.zeros_like(u)

    for i in range(np.shape(vitesse_n)[0]) :
        #vitesse_t[i,:,:] = u[i,:,:]*vec_t[0][i] + v[i,:,:]*vec_t[1][i]    
        vitesse_n[i,:,:] = u[i,:,:]*vec_n[0][i] + v[i,:,:]*vec_n[1][i]

    cs_ds["vitesse transversale"] = (("cross_section_idx", "month", "depth"), vitesse_n)

    return cs_ds



def get_climate_index():
    idx_names = ['NAO', 'AO', 'PDO', 'SOI']
    time = pd.date_range(start='2003-01-01', end='2023-12-01', freq='MS')

    dss = []
    

    for name in idx_names:
        df = pd.read_csv(f'data/{name}.csv', delim_whitespace=True, skiprows=0, names=['Year', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        df_melted = df.melt(id_vars=['Year'], var_name='Month', value_name='Value')
        df_melted['Date'] = pd.to_datetime(df_melted[['Year', 'Month']].assign(DAY=1))
        df_melted = df_melted.sort_values('Date').set_index('Date')
        df_melted = df_melted.drop(columns=['Year', 'Month'])
        
        ds = xr.Dataset.from_dataframe(df_melted).rename_dims({'Date': 'time'}).rename_vars({'Value': name, 'Date': 'time'})
        dss.append(ds)

    ds = xr.merge(dss)
    ds = ds.sel(time=slice('2003', '2023'))
    return ds



def plot2(Mar,Sept,contMar,contSept,colour1,colour2, vmin ,vmax, name, extent = [-180,180, 70, 90] ): 
    proj = ccrs.NorthPolarStereo()
    proj_og = ccrs.PlateCarree()     

    ds = load_climatology_with_deptho()
    lon = ds.longitude
    lat = ds.latitude  
    # plot pour la fin d'hivers et la fin d'été vue d'en haut 

    fig, ax = plt.subplots(1, 2, subplot_kw={'projection': proj}, figsize=(18, 18))

    # Flatten the axes array for easier iteration
    ax = ax.flatten()

    # Plot the first subplot
    pc = ax[0].pcolormesh(lon, lat, Mar, transform=proj_og, cmap=colour1, vmin = vmin, vmax = vmax)
    contour = ax[0].contour(lon, lat, contMar, levels=[0.15, 0.5, 0.85], colors=colour2, linewidths=1, transform=proj_og)
    ax[0].clabel(contour, inline=True, fontsize=8)
    ax[0].gridlines()
    ax[0].set_extent(extent, crs=proj_og)
    ax[0].set_title(f'{name} at the end of Boreal Winter')
    ax[0].coastlines()

    # Plot the second subplot
    pc = ax[1].pcolormesh(lon, lat, Sept, transform=proj_og, cmap=colour1, vmin = vmin, vmax = vmax)
    contour = ax[1].contour(lon, lat, contSept, levels=[0.15, 0.5, 0.85], colors=colour2, linewidths=1, transform=proj_og)
    ax[1].clabel(contour, inline=True, fontsize=8)
    ax[1].set_extent(extent, crs=proj_og)
    ax[1].set_title(f'{name} at the end of Boreal Summer')
    ax[1].coastlines()
    ax[1].gridlines()

    fig.suptitle(f'{name} Comparison at the End of Boreal Winter and Summer', fontsize=16,x = 0.52, y = 0.71)
    
    # Define the position for the colorbar [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.92, 0.32, 0.04, 0.35])  # Adjust these values as needed
    cbar = fig.colorbar(pc, cax=cbar_ax, orientation='vertical', label='Velocity [m/s]')
    cbar.set_label(f'{name}', fontsize=12)

    plt.show()




def plot2t(Mar,Sept,contMar,contSept,colour1,colour2, vmin ,vmax, name, U3, V3, U9 ,V9 ,namequiv = 'wind', extent = [-180,180, 70, 90] ): 
    proj = ccrs.NorthPolarStereo()
    proj_og = ccrs.PlateCarree()     

    ds = load_climatology_with_deptho()
    lon = ds.longitude
    lat = ds.latitude  

    if namequiv == 'wind' :
        i = 3
    else :
        i = 20

    # plot pour la fin d'hivers et la fin d'été vue d'en haut 

    fig, ax = plt.subplots(1, 2, subplot_kw={'projection': proj}, figsize=(18, 18))

    # Flatten the axes array for easier iteration
    ax = ax.flatten()

    # Plot the first subplot
    pc = ax[0].pcolormesh(lon, lat, Mar, transform=proj_og, cmap=colour1, vmin = vmin, vmax = vmax)
    contour = ax[0].contour(lon, lat, contMar, levels=[0.15, 0.5, 0.85], colors=colour2, linewidths=1, transform=proj_og)
    ax[0].clabel(contour, inline=True, fontsize=8)
    ax[0].gridlines()
    ax[0].set_extent(extent, crs=proj_og)
    ax[0].set_title(f'{name} at the end of Boreal Winter')
    ax[0].coastlines()


    #  Varying line width along a streamline
    Mag3 = np.sqrt(U3**2 + V3**2)
    lw = i *Mag3 / Mag3.max()
    ax[0].streamplot(lon.values,lat.values, U3.values,V3.values, density=[2, 2], color='k',transform=proj_og, linewidth=lw.values)


    # Plot the second subplot
    pc = ax[1].pcolormesh(lon, lat, Sept, transform=proj_og, cmap=colour1, vmin = vmin, vmax = vmax)
    contour = ax[1].contour(lon, lat, contSept, levels=[0.15, 0.5, 0.85], colors=colour2, linewidths=1, transform=proj_og)
    ax[1].clabel(contour, inline=True, fontsize=8)
    ax[1].set_extent(extent, crs=proj_og)
    ax[1].set_title(f'{name} at the end of Boreal Summer')
    ax[1].coastlines()
    ax[1].gridlines()

    #  Varying line width along a streamline
    Mag9 = np.sqrt(U9**2 + V9**2)
    lw = i *Mag9 / Mag9.max()
    ax[1].streamplot(lon.values,lat.values, U9.values,V9.values, density=2, color='k',transform=proj_og, linewidth=lw.values)


    fig.suptitle(f'{name} Comparison at the End of Boreal Winter and Summer with ' + f'{namequiv} streamlines', fontsize=16,x = 0.52, y = 0.71)
    
    # Define the position for the colorbar [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.92, 0.32, 0.04, 0.35])  # Adjust these values as needed
    cbar = fig.colorbar(pc, cax=cbar_ax, orientation='vertical', label='Velocity [m/s]')
    cbar.set_label(f'{name}', fontsize=12)

    plt.show()



def properties(ds, zone):
    if ds.month == 9:
        name = 'summer'
    else:
        name = 'winter'
    fig, axs = plt.subplots(6, 1, figsize=(15, 15))
    axs.flatten()
    
    # Temperature
    pc = axs[0].pcolormesh(ds.dist, ds.depth, ds['T'].values.T, cmap=cmo.thermal)
    plt.colorbar(pc, ax=axs[0], label='Temperature [°C]')
    axs[0].set_title('Temperature cross-section')

    # Salinity
    pc = axs[1].pcolormesh(ds.dist, ds.depth, ds['S'].values.T, cmap=cmo.haline)
    plt.colorbar(pc, ax=axs[1], label='Salinity [PSU]')
    axs[1].set_title('Salinity cross-section')

    # Density
    ds['rho'] = xr.apply_ufunc(gsw.rho_t_exact, ds['S'], ds['T'], ds.depth, dask='parallelized', output_dtypes=[float])
    pc = axs[2].pcolormesh(ds.dist, ds.depth, ds['rho'].values.T, cmap=cmo.dense)
    plt.colorbar(pc, ax=axs[2], label='Density [kg/m³]')
    axs[2].set_title('Density cross-section')

    cmap_min = -0.1
    cmap_max = 0.1

    # Vitesse transversale
    pc = axs[3].pcolormesh(ds.dist, ds.depth, (ds['V'] * ds.normal_meridional + ds['U'] * ds.normal_zonal).values.T, cmap='seismic', vmin=cmap_min, vmax=cmap_max)
    plt.colorbar(pc, ax=axs[3], label='m/s')
    axs[3].set_title('Vitesse transversale cross-section')
   
    # V
    pc = axs[4].pcolormesh(ds.dist, ds.depth, ds['V'].values.T, cmap='seismic', vmin=cmap_min, vmax=cmap_max)
    plt.colorbar(pc, ax=axs[4], label='m/s')
    axs[4].set_title('Meridional transport')

    # U
    pc = axs[5].pcolormesh(ds.dist, ds.depth, ds['U'].values.T, cmap='seismic', vmin=cmap_min, vmax=cmap_max)
    plt.colorbar(pc, ax=axs[5], label='m/s')
    axs[5].set_title('Zonal transport ')

    for ax in axs:
        ax.plot(ds.dist, ds['deptho'])
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('Depth [m]')
        ax.set_ylim(0, 500)
        ax.invert_yaxis()
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(14)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(f'Properties of the {zone} at the end of boreal {name}', fontsize=20)
    

    plt.show()
