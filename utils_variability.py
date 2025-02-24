import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmocean.cm as cmo
import cartopy.crs as ccrs
import gsw
from eofs.xarray import Eof
from scipy.fft import fft, fftfreq
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


from utils import *


def plot_distribution(da, bins=100, **kwargs):

    fig, ax= plt.subplots(1, 1, figsize=(6, 4))
    
    all_values = da.values.flatten()
    ax.hist(all_values, bins=100, alpha=0.5, label='SIC', density=True)
    mu, sigma = np.nanmean(all_values), np.nanstd(all_values)
    x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
    ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2)), label='Normal Approximation')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {da.long_name}')
    ax.legend()
    plt.show()


def fit_eofs(da):
    weights = np.sqrt(np.cos(np.deg2rad(da.latitude))) + da.longitude * 0
    solver = Eof(da, weights=weights)
    return solver

def fast_eof_analysis(eof_solver, da):
    eof_solver.varianceFraction().plot(marker='o')
    plt.xlim(0, 20)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()})
    pcm = ax.pcolormesh(da.longitude, da.latitude, da.std('time'), transform=ccrs.PlateCarree(), cmap='YlGn')
    ax.coastlines()
    plt.colorbar(pcm, ax=ax, label=f'{da.units}')
    ax.set_title(f'{da.long_name} standard deviation')
    plt.show()

    fig, axs = plt.subplots(2, 3, figsize=(20, 15), subplot_kw={'projection': ccrs.NorthPolarStereo()})

    axs = axs.flatten()

    max_val = np.abs(eof_solver.eofs(eofscaling=2)[:6]).max()

    for i in range(6):
        ax = axs[i]
        ax.coastlines()
        ax.gridlines()
        pcm = ax.pcolormesh(
            eof_solver.eofs().longitude, 
            eof_solver.eofs().latitude, 
            eof_solver.eofs(eofscaling=2)[i], 
            transform=ccrs.PlateCarree(), 
            cmap=cmo.balance,
            vmin=-max_val, 
            vmax=max_val
        )
        ax.set_title(f'EOF {i+1}')
    plt.colorbar(pcm, ax=axs, orientation='horizontal', shrink=0.8, label=da.units)
    plt.suptitle(f'{da.long_name} EOFs')
    plt.show()



    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for i in range(3):
        ax.plot(eof_solver.pcs(1).time, eof_solver.pcs(1)[:, i], label=f'PC {i+1}')
    plt.legend()
    plt.title(f'{da.long_name} PCs (time domain)')
    plt.show()


    # Number of sample points
    N = eof_solver.pcs().shape[0]
    # Sample spacing
    T = (eof_solver.pcs(1).time[1] - eof_solver.pcs(1).time[0]).values.astype('timedelta64[D]').astype(int) / 365.25

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for i in range(3):
        yf = fft(eof_solver.pcs(1)[:, i].values)
        xf = fftfreq(N, T)[:N//2]
        ax.plot(xf, 2.0/N * np.abs(yf[:N//2]), label=f'PC {i+1}')

    ax.axvline(x=1, color='r', linestyle='--', label='f=1/year')
    plt.legend()
    plt.title(f'{da.long_name} PCs (frequency domain)')
    plt.xlabel('Frequency [1/year]')
    plt.ylabel('Amplitude')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()



def project_on_field(anomalies, eof_solver):
    projection = (anomalies * eof_solver.pcs(1, 6)).mean('time')

    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()})
    pcm = ax.pcolormesh(anomalies.longitude, anomalies.latitude, anomalies.std('time'), transform=ccrs.PlateCarree(), cmap='YlGn')
    ax.coastlines()
    plt.colorbar(pcm, ax=ax, label=f'{anomalies.units}')
    ax.set_title(f'{anomalies.long_name} standard deviation')
    plt.show()


    fig, axs = plt.subplots(2, 3, figsize=(20, 15), subplot_kw={'projection': ccrs.NorthPolarStereo()})

    axs = axs.flatten()

    max_val = np.abs(projection).max()

    for i in range(6):
        ax = axs[i]
        ax.coastlines()
        ax.gridlines()
        pcm = ax.pcolormesh(
            projection.longitude, 
            projection.latitude, 
            projection.sel(mode=i), 
            transform=ccrs.PlateCarree(), 
            cmap=cmo.balance,
            vmin=-max_val,
            vmax=max_val
        )
        ax.set_title(f'EOF {i+1}')
    plt.colorbar(pcm, ax=axs, label=f'{anomalies.units}', orientation='horizontal', shrink=0.8)
    plt.suptitle(f'{anomalies.long_name} projection on EOFs')
    plt.show()

