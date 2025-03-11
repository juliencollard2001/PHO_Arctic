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

def mise_en_forme_daily(ds) :
    #On met en forme le dataset pour qu'il soit plus facile à manipuler
    #On renomme les variables
    ds = ds.rename_vars({
        't2m': 't2m',
        'u10': 'u10',
        'v10': 'v10',
        'slhf': 'slhf',
        'ssr': 'ssr',
        'str': 'str',
        'sshf': 'sshf',
    })
    #On ne garde que les variables qui nous intéressent
    ds = ds[['t2m', 'u10', 'v10', 'slhf', 'ssr', 'str', 'sshf']]

    #moyenne sur latitude et longitude
    ds = ds.mean(dim = ['latitude', 'longitude'])

    #On convertit les variables en numpy array
    t2m = ds['t2m'].values
    u10 = ds['u10'].values
    v10 = ds['v10'].values
    slhf = ds['slhf'].values
    ssr = ds['ssr'].values
    str = ds['str'].values
    sshf = ds['sshf'].values
    return t2m, u10, v10, slhf, ssr, str, sshf

def mise_en_forme_snow(ds_snow) :
    #On met en forme le dataset pour qu'il soit plus facile à manipuler
    #On renomme les variables
    ds_snow = ds_snow.rename_vars({
        'sd': 'sd',
    })
    #On ne garde que les variables qui nous intéressent
    ds_snow = ds_snow[['sd']]

    #moyenne sur latitude et longitude
    ds_snow = ds_snow.mean(dim = ['latitude', 'longitude'])

    #On convertit les variables en numpy array
    snow_depth = ds_snow['sd'].values
    return snow_depth

def surface_flux_downwards(
        sw, #shortwave, J/m2 descendant
        lw, #longwave, J/m2 upwards
        latent, #latent heat, J/m2 downwards
        sensible, #sensible heat, J/m2 downwards
) :
    accumulation_period = 3600 #en secondes, donnee par dataset
    flux = sw + lw + latent + sensible # J/m2
    flux = flux/accumulation_period #W/m2
    return flux

def ice_growth_rate(
        epaisseur, #epaisseur
        temp_air,
        fs,
        epaisseur_snow,
        freezing_temp = -1.8,
        snow = False, #boolean
    ) :

    #flux compté positivement lorsque la glasse grossit
    #pas de stockage de chaleur dans la glace

    lambda_snow = 0.3
    lambda_ice = 2.1  #conductivite thermique de la glace
    rho = 917   #Masse volumique de la glace en kg/m3
    Latent = 333000   #Chaleur latente fusion glace en J/kg

    if snow :
        temp_top = temp_air
    else :
        temp_top = temp_air
    
    #Basal growth
    if temp_top < freezing_temp :
        if epaisseur == 0 :
            rate_basal = 0 #valeur paramétrique pour ne pas que ça diverge
        else :
            rate_basal = -1*(temp_top - freezing_temp)/(rho*Latent*((epaisseur/lambda_ice) + (epaisseur_snow/lambda_snow)))    #On néglige la convection 
    else :
        rate_basal = 0
    
    #Top growth

    #Top melt
    rate_top = 0
    if fs > 0 :
        if epaisseur_snow < 0.001 : #Epaisseur snow suffisamment faible (non prise en compte ici)
            if temp_top > freezing_temp :
                rate_top = (-1/(rho*Latent))*fs   #Va dans le sens d'une compensation de la croissance de la glace, lorsqu'elle existe
        

    #Total growth/melt (positive when growth)
    rate = rate_basal + rate_top
    return rate

def ice_thickness_th(
        epaisseur,  #profil à temps initial
        t2m,    #np.array 
        fs, #flux surface
        snow_depth,
        duree, #en jours, valeur par défaut 1 an
        res = 2 #nombre de points/jours à calculer par jour
    ) :
    #schema d'ordre 1
    # Définition des variables
    deltat = 1/res # en jours/points
    N = int(res*duree)

    epaisseur_init = epaisseur[0]

    thickness = np.ones(N)
    thickness[:] = 0

    time = np.zeros(N)

    thickness[0] = epaisseur_init

    for i in range(1,N) :

        if i < len(snow_depth) :
            thickness[i] = thickness[i - 1] + ice_growth_rate(thickness[i-1], t2m[i], fs[i], snow_depth[i])*deltat*86400   #Pour convertir deltat en secondes
        else :
            thickness[i] = thickness[i - 1] + ice_growth_rate(thickness[i-1], t2m[i], fs[i], 0)*deltat*86400
        if thickness[i] < 0 :
            thickness[i] = 0   #On ne peut pas avoir une épaisseur négative
        if i > 0 :
            time[i] = time[i - 1] + deltat
        if (i%(2*(365)) - 2*(365 - 365 // 6)) == 0:
            thickness[i] = epaisseur[i]
            thickness[i - 1] = np.nan
        #if (i%(2*(365)) - 2*(365 - 365//2 - 365 // 6)) == 0 :
        #    thickness[i] = np.nan
    
    return time, thickness

def permutation_circ_mois(array, mois_initial_v1, mois_initial_v2) :

    #But : arranger un array JFMAMJ ... en MAMJ... par exemple

    incr = mois_initial_v2 - mois_initial_v1
    
    array_permut = np.zeros_like(array)

    for i in range(12) :
        if i + incr < 12 :
            array_permut[i] = array[i + incr]
        elif i + incr >= 12 :
            array_permut[i] = array[i + incr - 12]

    return array_permut

def choose_coords(coord_lat, coord_lon, lats, lons) :

    #Renvoie les index du tableau à prendre pour telle longitude/latitude

    index_coord_lat = np.argmin(np.abs(lats - coord_lat))
    index_coord_lon = np.argmin(np.abs(lons - coord_lon))

    return index_coord_lat, index_coord_lon



def y_obsVSmod(ds, mois_depart, coord1, coord2) :
    sit_obs = ds['SIT'][:,coord1,coord2]

    temp_2m = ds['t2m'][:,coord1,coord2] - 273.15

    temp_2m_clim = np.zeros_like(sit_obs)
    for i in range(np.shape(sit_obs)[0]) :
        temporary = []
        for j in range(temp_2m.shape[0]) :
            if (j - i)%12 == 0 :
                temporary.append(temp_2m[j])
        temporary = np.array(temporary)
        temp_2m_clim[i] = np.mean(temporary)

    print(temp_2m_clim)

    y_obs = permutation_circ_mois(sit_obs, 0, mois_depart) 
    temp_permut_clim = permutation_circ_mois(temp_2m_clim, 0, mois_depart) 

    sit_mod = ice_thickness_th(y_obs[0], temp_permut_clim)
    time = ds['month']

    y_mod = np.ones_like(sit_obs)

    for i in range(12) :
        y_mod[i] = sit_mod[30*i]

    return y_obs, y_mod