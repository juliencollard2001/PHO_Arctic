from utils import *

def cross_section_nodepth(
     ds: xr.Dataset, 
        lon1: float, 
        lat1: float, 
        lon2: float, 
        lat2: float, 
        N : int = 50,
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


def vecteurs_coupe_air(cs_dsair : xr.Dataset) -> xr.Dataset:

    # PARTIE DÉTERMINATION VECTEURS UNITAIRES POUR DECRIRE LA SECTION
        
    dist = cs_dsair['dist']
    lat = cs_dsair['latitude']
    lon = cs_dsair['longitude']

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

    u = cs_dsair.U
    v = cs_dsair.V

    #vitesse_t = np.zeros_like(u)
    vitesse_n = np.zeros_like(u)

    print(vitesse_n.shape)

    for i in range(np.shape(vitesse_n)[0]) :
        #vitesse_t[i,:,:] = u[i,:]*vec_t[0][i] + v[i,:]*vec_t[1][i]    
        vitesse_n[i,:] = u[i,:]*vec_n[0][i] + v[i,:]*vec_n[1][i]

    cs_dsair["vitesse transversale"] = (("cross_section_idx", "month"), vitesse_n)

    return cs_dsair


def isoligne_cross_section(
    cs_ds: xr.Dataset,
    variable: str,
    level: float,
    ) -> xr.Dataset:

    isoligne = xr.Dataset()
    isoligne = isoligne.assign_coords(cross_section_idx=cs_ds.cross_section_idx)
    isoligne = isoligne.assign_coords(month=cs_ds.month)    
    #isoligne = isoligne.assign_coords(latitude=cs_ds.latitude)
    #isoligne = isoligne.assign_coords(longitude=cs_ds.longitude)
    #isoligne = isoligne.assign_coords(dist=cs_ds.dist)

    depth_ref = cs_ds['depth'].values #Comme un étalonnage
    

    index = cs_ds['cross_section_idx'].values

    index_depth = np.zeros((len(index), len(cs_ds['month'])))
    depth = np.zeros_like(index_depth, dtype= np.float32)

    for i in range(len(index)) :
        for j in range(len(isoligne['month'])) :
            inter = np.argmin(np.abs(cs_ds[variable][i,j].values - level))
            index_depth[i,j] = inter
            depth[i,j] = depth_ref[inter]

    isoligne['depth'] = (("cross_section_idx", "month"), depth)

    return isoligne