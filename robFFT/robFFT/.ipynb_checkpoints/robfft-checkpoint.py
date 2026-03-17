import xarray as xr
import numpy as np
import pylops

def compute_fft2(da, dim_y='y', dim_x='x', tapering=False):
    """
    Compute the 2D Fourier transform of a 2D xarray.DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        Input 2D data array.
    dim_y, dim_x : str
        Names of the two spatial dimensions in `da`.

    Returns
    -------
    xr.DataArray
        The shifted 2D FFT, with new coordinate dimensions 'ky' and 'kx' 
        representing the Fourier frequencies.
    """

    if tapering == True:
        tapewidth = 0.1
        nmask = (da[dim_x].size, da[dim_y].size)
        da = da - da.mean() # remove mean
        ntap  = (int(tapewidth*da[dim_x].size),int(tapewidth*da[dim_y].size))
        tape  = pylops.utils.tapers.tapernd(nmask, ntap, "hanning")
        da_taped = da*np.transpose(tape)
    else:
        da_taped = da
        
    # 1) 2D FFT
    fft = xr.apply_ufunc(
        np.fft.fft2, da_taped,
        input_core_dims=[[dim_y, dim_x]],
        output_core_dims=[['ky', 'kx']],
        vectorize=True,
        dask='parallelized',      # if DataArray is backed by dask
        output_dtypes=[complex]
    )

    # 2) Shift zero-frequency to center
    fft_shifted = xr.apply_ufunc(
        np.fft.fftshift, fft,
        input_core_dims=[['ky', 'kx']],
        output_core_dims=[['ky', 'kx']],
        vectorize=True
    )

    # 3) Build frequency coordinates
    N_y, N_x = da.sizes[dim_y], da.sizes[dim_x]
    dy = float(da[dim_y][1] - da[dim_y][0])
    dx = float(da[dim_x][1] - da[dim_x][0])
    freq_y = np.fft.fftshift(np.fft.fftfreq(N_y, d=dy))*2*np.pi
    freq_x = np.fft.fftshift(np.fft.fftfreq(N_x, d=dx))*2*np.pi

    # 4) Assign frequency coords back onto the DataArray
    fft_shifted = fft_shifted.assign_coords({
        'ky': freq_y,
        'kx': freq_x
    })
    
    return fft_shifted

def compute_omega2(N2,f2,k2,l2,m2,Temp,z):

    Me = 5.9722*1e24  #[kg]
    G  = 6.6743*1e-11 #[m^3/kg/s^2]
    Re = 6.371*1e6    #[m]
    cp = 1003.5       #[J/kg/K]
    
    # Unit detection: Assume km if dz < 10, else meters
    if z < 100:  
        z *= 1e3           # Convert dz to meters

    g  = G*Me/(Re+z)**2   #[m/s^2]
    R  = 8.314            #[J/mol/K]
    M  = 28.949*1e-3      #[kg/mol]
    H  = R * Temp / M / g #[m]
    alpha2 = 1 / 4 / H**2 #[1/m^2]
    omega2 = (N2 * (k2 + l2) + f2 * (m2 + alpha2)) / (k2 + l2 + m2 + alpha2) #[1/s^2]  

    return omega2

def compute_N2(Temp1,Temp2,z1,z2):
    """
    Compute the thermal stratification based on the local temperature gradient.

    Parameters
    ----------
    Temp1, Temp2 : float
        Input two temperature values separated by dz. 
    z1, z2 : float
        Input two z values 

    Returns
    -------
    float
        squared Brunt Väisälä frequency.
    """
    
    Me = 5.9722*1e24
    G  = 6.6743*1e-11
    Re = 6.371*1e6
    cp = 1003.5 #[J/kg/K]
    
    dz = z2 - z1             #[m]
    z_mean = (z1+z2)/2       #[m]
    T_mean = (Temp1+Temp2)/2 #[K]

    # Unit detection: Assume km if dz < 10, else meters
    if np.abs(dz) < 10:  
        dz *= 1e3           # Convert dz to meters
        z_mean*=1e3
    
    dT_dz = (Temp2-Temp1) / dz       #[K/m]
    g         = G*Me/(Re+z_mean)**2  #[m/s^2]
    N2        = g/T_mean*(dT_dz+g/cp)#[1/s^2]

    return N2

def compute_vertical_wavenumber_and_intrinsic_frequency(ds,var,dim_x='x',dim_y='y',dim_z='z',tapering=False,lat_mean=45):
    """
    Compute the vertical wavenumber (m) from 2D+1-FFT in z-direction.
    Use dispersion relation to infer intrinsic frequency (omega).

    Parameters
    ----------
    ds : xarray.DataSet
        Input 3D data array containing wave structures to be analyzed.
        Input 3D Temp data array to compute stratification (N). 
        
    var : str
        Name of the atmospheric variable that is to be analyzed.
        
    dim_x, dim_y, dim_z : str
        Names of the spatial dimensions in `ds`.

    tapering : logical
        logical whether tapering should be applied at the edges. Default: False
    
    lat_mean : float
        center latitude for calculation of coriolis frequency. Default: 45

    Returns
    -------
    xr.DataSet
        Contains spectral power, vertical wavenumber, and intrinsic frequency as functions of 'kx', 'ky', and 'z'.
    """

    wavefield = ds[var]
    Temp      = ds['temp_bg'].mean(dim=(dim_x,dim_y)) #[K]
    Uwind     = ds['u_bg'].mean(dim=(dim_x,dim_y))    #[m/s]
    Vwind     = ds['v_bg'].mean(dim=(dim_x,dim_y))    #[m/s]
    f2        = (4*np.pi/(24*3_600)*np.sin(np.deg2rad(lat_mean)))**2 #[1/s]

    fft2_test  = compute_fft2(wavefield.isel({dim_z: 0}),dim_x=dim_x,dim_y=dim_y, tapering=tapering)
    KX, KY     = np.meshgrid(fft2_test['kx'].values, fft2_test['ky'].values, indexing='ij')  #[1/m]
    m          = np.empty((fft2_test.coords['kx'].size,fft2_test.coords['ky'].size,ds.coords[dim_z].size-1))
    omega2     = np.empty((fft2_test.coords['kx'].size,fft2_test.coords['ky'].size,ds.coords[dim_z].size-1))
    omegaEx    = np.empty((fft2_test.coords['kx'].size,fft2_test.coords['ky'].size,ds.coords[dim_z].size-1))
    fft2_X_ABS = np.empty((fft2_test.coords['kx'].size,fft2_test.coords['ky'].size,ds.coords[dim_z].size-1))

    z_new = np.convolve(ds.coords[dim_z].values, [0.5, 0.5], mode='valid')

    # Pay attention: Running along alt means from top to bottom!
    for iz in range(ds.coords[dim_z].size-1):
        dz             = np.abs(ds.coords[dim_z].isel({dim_z: iz+1}) - ds.coords[dim_z].isel({dim_z: iz}))
        N2             = compute_N2(Temp.isel({dim_z: iz}).values,Temp.isel({dim_z: iz+1}).values,Temp[dim_z].isel({dim_z: iz}).values,Temp[dim_z].isel({dim_z: iz+1}).values) #[1/s^2]
        fft2_result    = compute_fft2(wavefield.isel({dim_z: iz}),dim_x=dim_x,dim_y=dim_y,tapering=tapering)
        fft2_result_dz = compute_fft2(wavefield.isel({dim_z: iz+1}),dim_x=dim_x,dim_y=dim_y,tapering=tapering)
        fft2_X         = fft2_result*np.conjugate(fft2_result_dz)
        fft2_X_ABS[:,:,iz]= np.abs(fft2_X).T
        fft2_X_PHA     = np.angle(fft2_X).T     #[rad]
        m[:,:,iz]      = fft2_X_PHA / dz.values #[rad/m]
        T_mean         = (Temp.isel({dim_z: iz})+Temp.isel({dim_z: iz+1}))/2 #[K]
        # intrinsic
        omega2[:,:,iz] = compute_omega2(N2,f2,KX**2,KY**2,m[:,:,iz]**2,T_mean.values,z_new[iz]) #[1/s^2]
        # observed
        omegaEx[:,:,iz] = np.sqrt(omega2[:,:,iz]) + KX*Uwind.isel({dim_z: iz}).values + KY*Vwind.isel({dim_z: iz}).values #[1/s]


    result = xr.Dataset(
        {
            'm': xr.DataArray(
                m,
                dims=('kx','ky',dim_z),
                coords={'kx': fft2_result['kx'], 'ky': fft2_result['ky'], dim_z: z_new},
            ),
            'omega': xr.DataArray(
                np.sqrt(omega2),
                dims=('kx','ky',dim_z),
                coords={'kx': fft2_result['kx'], 'ky': fft2_result['ky'], dim_z: z_new},
            ),
            'omega_shifted': xr.DataArray(
                omegaEx,
                dims=('kx','ky',dim_z),
                coords={'kx': fft2_result['kx'], 'ky': fft2_result['ky'], dim_z: z_new},
            ),
            'spectral_power': xr.DataArray(
                fft2_X_ABS,
                dims=('kx','ky',dim_z),
                coords={'kx': fft2_result['kx'], 'ky': fft2_result['ky'], dim_z: z_new},
            ),

        }
    )

    return result


def compute_momentum_flux(ds,dim_x='x',dim_y='y',dim_z='z', tapering=False):
    """
    Compute the momentum flux u'w' in spectral space from 2D-FFT

    Parameters
    ----------
    ds : xarray.DataSet
        Input 3D data array containing wave structures to be analyzed.
        Input 3D Temp data array to compute stratification (N).
        Assumes data contains vertical wind 'w' and zonal wind 'u'

    dim_x, dim_y, dim_z : str
        Names of the spatial dimensions in `ds`.

    Returns
    -------
    xr.DataSet
        Contains momentum flux as functions of 'kx', 'ky', and 'z'.
    """

    fft2_test = compute_fft2(ds['w'].isel({dim_z: 0}),dim_x=dim_x,dim_y=dim_y,tapering=tapering)
    uw        = np.empty((fft2_test.coords['kx'].size,fft2_test.coords['ky'].size,ds.coords[dim_z].size))

    for iz in range(ds.coords[dim_z].size):
        fft2_result_w = compute_fft2(ds['w'].isel({dim_z: iz}),dim_x=dim_x,dim_y=dim_y,tapering=tapering)
        fft2_result_u = compute_fft2(ds['u'].isel({dim_z: iz}),dim_x=dim_x,dim_y=dim_y,tapering=tapering)
        uw[:,:,iz]    = np.real(fft2_result_w*np.conjugate(fft2_result_u)).T

    z_new_half = np.convolve(ds.coords[dim_z].values, [0.5, 0.5], mode='valid')
    uw_half    = np.empty((fft2_test.coords['kx'].size,fft2_test.coords['ky'].size,ds.coords[dim_z].size-1))
    for iz in range(ds.coords[dim_z].size-1):
        uw_half[:,:,iz] = (uw[:,:,iz]+uw[:,:,iz+1])/2

    result = xr.Dataset(
        {
            'uw': xr.DataArray(
                uw_half,
                dims=('kx','ky',dim_z),
                coords={'kx': fft2_result_u['kx'], 'ky': fft2_result_u['ky'], dim_z: z_new_half},
            ),
        }
    )

    return result


def compute_energy_flux(ds,dim_x='x',dim_y='y',dim_z='z', tapering=False):
    """
    Compute the energy flux p'w'  in spectral space from 2D-FFT

    Parameters
    ----------
    ds : xarray.DataSet
        Input 3D data array containing wave structures to be analyzed.
        Input 3D Temp data array to compute stratification (N).
        Assumes data contains vertical wind 'w' and pressure 'pres'

    dim_x, dim_y, dim_z : str
        Names of the spatial dimensions in `ds`.

    Returns
    -------
    xr.DataSet
        Contains energy flux as functions of 'kx', 'ky', and 'z'.
    """

    fft2_test  = compute_fft2(ds['w'].isel({dim_z: 0}),dim_x=dim_x,dim_y=dim_y, tapering=tapering)
    pw          = np.empty((fft2_test.coords['kx'].size,fft2_test.coords['ky'].size,ds.coords[dim_z].size))

    for iz in range(ds.coords[dim_z].size):
        fft2_result_w = compute_fft2(ds['w'].isel({dim_z: iz}),dim_x=dim_x,dim_y=dim_y,tapering=tapering)
        fft2_result_p = compute_fft2(ds['pres'].isel({dim_z: iz}),dim_x=dim_x,dim_y=dim_y,tapering=tapering)
        pw[:,:,iz]    = np.real(fft2_result_p*np.conjugate(fft2_result_w)).T

    z_new_half = np.convolve(ds.coords[dim_z].values, [0.5, 0.5], mode='valid')
    pw_half    = np.empty((fft2_test.coords['kx'].size,fft2_test.coords['ky'].size,ds.coords[dim_z].size-1))
    for iz in range(ds.coords[dim_z].size-1):
        pw_half[:,:,iz] = (pw[:,:,iz]+pw[:,:,iz+1])/2

    result = xr.Dataset(
        {
            'pw': xr.DataArray(
                pw_half,
                dims=('kx','ky',dim_z),
                coords={'kx': fft2_result_w['kx'], 'ky': fft2_result_w['ky'], dim_z: z_new_half},
            ),
        }
    )

    return result

def remove_background(
    ds,
    variables=None,
    dim_lat="latitude",
    dim_lon="longitude",
    wl_cutoff=2000,
    bg_suffix="_bg",
):
    """
    Split selected variables in an xr.Dataset into
      1) perturbation / small-scale component (stored under original name)
      2) background / large-scale component (stored as <var><bg_suffix>)
    using a 1D FFT along longitude.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    variables : list of str or None
        Names of variables to split into perturbation and background.
        If None, all variables containing dim_lon are processed.
    dim_lat : str
        Name of latitude dimension.
    dim_lon : str
        Name of longitude dimension.
    wl_cutoff : float
        Cutoff wavelength in km. Wavelengths >= wl_cutoff go into the
        background component.
    bg_suffix : str
        Suffix for background variables, e.g. "_bg".

    Returns
    -------
    xr.Dataset
        Dataset where selected variables are replaced by their perturbation
        component, and additional background variables are added as
        '<name><bg_suffix>'.
        Unselected variables are copied unchanged.
    """
    r_e = 6378.0  # Earth radius in km

    if not isinstance(ds, xr.Dataset):
        raise TypeError("This function expects an xr.Dataset.")
    if dim_lon not in ds.dims:
        raise ValueError(f"Longitude dimension '{dim_lon}' not found.")
    if dim_lat not in ds.dims:
        raise ValueError(f"Latitude dimension '{dim_lat}' not found.")
    if ds.sizes[dim_lon] < 2:
        raise ValueError(f"Need at least 2 points along '{dim_lon}'.")
    if wl_cutoff <= 0:
        raise ValueError("wl_cutoff must be > 0.")

    lon = ds[dim_lon]
    lat = ds[dim_lat]

    # Check approximately uniform longitude spacing
    dlon = lon.diff(dim_lon)
    if not np.allclose(dlon, dlon.isel({dim_lon: 0}), rtol=1e-3):
        raise ValueError(f"'{dim_lon}' must be uniformly spaced for FFT filtering.")

    dx = np.deg2rad(float(dlon.isel({dim_lon: 0})))  # radians
    nlon = ds.sizes[dim_lon]

    # Decide which variables to process
    if variables is None:
        variables_to_filter = [v for v in ds.data_vars if dim_lon in ds[v].dims]
    else:
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise ValueError(f"Variables not found in dataset: {missing}")

        no_lon = [v for v in variables if dim_lon not in ds[v].dims]
        if no_lon:
            raise ValueError(
                f"These variables do not contain longitude dimension '{dim_lon}': {no_lon}"
            )

        variables_to_filter = list(variables)

    # Start with unchanged copy of full dataset
    out = ds.copy()

    if len(variables_to_filter) == 0:
        return out

    # Only process requested variables
    ds_sel = ds[variables_to_filter]

    # Forward real FFT along longitude
    spec = xr.apply_ufunc(
        np.fft.rfft,
        ds_sel,
        input_core_dims=[[dim_lon]],
        output_core_dims=[["kx"]],
        dask="parallelized",
        output_dtypes=[np.complex128],
        dask_gufunc_kwargs={"output_sizes": {"kx": nlon // 2 + 1}},
        kwargs={"axis": -1},
    )

    kx = np.fft.rfftfreq(nlon, d=dx)  # cycles per radian
    spec = spec.assign_coords(kx=("kx", kx))

    # Latitude-dependent cutoff in cycles per radian
    cutoff = (r_e * np.cos(np.deg2rad(lat))) / wl_cutoff

    # Keep wavelengths shorter than wl_cutoff in perturbation field
    mask_high = np.abs(spec["kx"]) > cutoff

    # Perturbation / small-scale component
    spec_high = spec.where(mask_high, 0)
    pert = xr.apply_ufunc(
        np.fft.irfft,
        spec_high,
        input_core_dims=[["kx"]],
        output_core_dims=[[dim_lon]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {dim_lon: nlon}},
        kwargs={"n": nlon, "axis": -1},
    )

    # Background / large-scale component
    spec_low = spec.where(~mask_high, 0)
    bg = xr.apply_ufunc(
        np.fft.irfft,
        spec_low,
        input_core_dims=[["kx"]],
        output_core_dims=[[dim_lon]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {dim_lon: nlon}},
        kwargs={"n": nlon, "axis": -1},
    )

    pert = pert.assign_coords({dim_lon: lon, dim_lat: lat})
    bg = bg.assign_coords({dim_lon: lon, dim_lat: lat})

    # Put perturbations back under original names, add background as new vars
    for var in variables_to_filter:
        out[var] = pert[var].transpose(*ds[var].dims)
        out[f"{var}{bg_suffix}"] = bg[var].transpose(*ds[var].dims)

        # preserve attrs
        out[var].attrs = ds[var].attrs.copy()
        out[f"{var}{bg_suffix}"].attrs = ds[var].attrs.copy()
        out[f"{var}{bg_suffix}"].attrs["description"] = (
            ds[var].attrs.get("description", var) + " background component"
        )

    out.attrs = ds.attrs.copy()

    return out

def lonlat_to_cartesian_grid(ds,
    lon="longitude", lat="latitude",
    nx=None, ny=None, dx_km=None, dy_km=None, lon0=None, lat0=None,
    method="linear", x_name="x", y_name="y",
    R_m=6371.0*1e3,
):
    """
    Regriddet ein xarray Dataset von einem (lon, lat)-Grid auf ein lokales
    kartesisches (x, y)-Grid in km.

    Annahmen
    --------
    - Die betrachtete lon/lat-Box ist relativ klein (~2000 km), sodass eine
      lokale Tangentialebenen-Näherung ausreichend ist.
    - Das ursprüngliche Grid ist rectilinear: lon und lat sind 1D-Koordinaten.
    - Geeignet für |lat| < ~80°.

    Parameter
    ---------
    ds : xr.Dataset oder xr.DataArray
        Eingabedaten mit lon- und lat-Koordinaten.
    lon, lat : str
        Namen der Längen- und Breitengrad-Koordinaten.
    nx, ny : int, optional
        Anzahl der Punkte im Zielgrid in x- bzw. y-Richtung.
        Falls nicht gesetzt, wird die Anzahl aus dem Eingang übernommen
        oder aus dx_km/dy_km berechnet.
    dx_km, dy_km : float, optional
        Zielauflösung in km. Falls gesetzt, werden nx/ny daraus bestimmt.
    lon0, lat0 : float, optional
        Mittelpunkt der Projektion. Falls None, wird der Mittelpunkt der Box
        aus den Koordinaten berechnet.
    method : str
        Interpolationsmethode für xarray.interp, z.B. "linear" oder "nearest".
    x_name, y_name : str
        Namen der kartesischen Zielkoordinaten.
    R_km : float
        Erdradius in km.

    Returns
    -------
    xr.Dataset oder xr.DataArray
        Dataset auf einem kartesischen Grid mit Dimensionen (y, x) statt (lat, lon).
        Zusätzlich werden 2D-Hilfskoordinaten `longitude_2d` und `latitude_2d`
        mitgegeben.
    """

    if lon not in ds.coords or lat not in ds.coords:
        raise ValueError(f"'{lon}' und/oder '{lat}' sind keine Koordinaten in ds.")

    lon_in = ds[lon].values
    lat_in = ds[lat].values

    if ds[lon].ndim != 1 or ds[lat].ndim != 1:
        raise ValueError("Diese Funktion erwartet 1D lon- und lat-Koordinaten.")

    # Hilfsfunktion: Longitude relativ zu Referenzpunkt unwrapen
    def wrap_relative(lon_vals, lon_ref):
        return lon_ref + ((lon_vals - lon_ref + 180.0) % 360.0 - 180.0)

    # Projektionzentrum bestimmen
    if lat0 is None:
        lat0 = float(np.mean(lat_in))

    if lon0 is None:
        lon_guess = float(np.mean(lon_in))
        lon_unwrapped_tmp = wrap_relative(lon_in, lon_guess)
        lon0 = float(np.mean(lon_unwrapped_tmp))

    # Longitude um lon0 herum unwrapen, damit sie monoton sortierbar ist
    lon_unwrapped = wrap_relative(lon_in, lon0)

    # Sortierung sicherstellen (wichtig für interp)
    lon_order = np.argsort(lon_unwrapped)
    lat_order = np.argsort(lat_in)

    ds_work = ds.isel({lon: lon_order, lat: lat_order}).assign_coords(
        {
            lon: lon_unwrapped[lon_order],
            lat: lat_in[lat_order],
        }
    )

    lon_sorted = ds_work[lon].values
    lat_sorted = ds_work[lat].values

    # Boxgröße im lokalen kartesischen System
    lat0_rad = np.deg2rad(lat0)
    cos_lat0 = np.cos(lat0_rad)
    if np.isclose(cos_lat0, 0.0):
        raise ValueError("cos(lat0) ist zu klein. Diese Approximation ist nahe der Pole ungeeignet.")

    x_min = R_m * cos_lat0 * np.deg2rad(lon_sorted.min() - lon0)
    x_max = R_m * cos_lat0 * np.deg2rad(lon_sorted.max() - lon0)
    y_min = R_m * np.deg2rad(lat_sorted.min() - lat0)
    y_max = R_m * np.deg2rad(lat_sorted.max() - lat0)

    # Zielauflösung / Zielgröße
    if dx_km is not None:
        nx = int(np.floor((x_max - x_min) / dx_km)) + 1
    elif nx is None:
        nx = ds.sizes[lon]

    if dy_km is not None:
        ny = int(np.floor((y_max - y_min) / dy_km)) + 1
    elif ny is None:
        ny = ds.sizes[lat]

    if nx < 2 or ny < 2:
        raise ValueError("nx und ny müssen jeweils >= 2 sein.")

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    # Zielgrid in km
    X, Y = np.meshgrid(x, y)

    lat_target = lat0 + np.rad2deg(Y / R_m)
    lon_target = lon0 + np.rad2deg(X / (R_m * cos_lat0))
    
    # numerisch robuste Begrenzung auf den gültigen Quellbereich
    eps_lon = 1e-10
    eps_lat = 1e-10
    
    lon_target = np.clip(lon_target, lon_sorted.min() + eps_lon, lon_sorted.max() - eps_lon)
    lat_target = np.clip(lat_target, lat_sorted.min() + eps_lat, lat_sorted.max() - eps_lat)

    # 2D Target-Coords als DataArrays für xarray.interp
    lon_da = xr.DataArray(
        lon_target,
        dims=(y_name, x_name),
        coords={y_name: y, x_name: x},
        name=lon,
    )
    lat_da = xr.DataArray(
        lat_target,
        dims=(y_name, x_name),
        coords={y_name: y, x_name: x},
        name=lat,
    )

    # Interpolation
    out = ds_work.interp({lon: lon_da, lat: lat_da}, method=method)

    # Hilfskoordinaten ergänzen
    out = out.assign_coords(
        {
            x_name: x,
            y_name: y,
        }
    )

    return out