import numpy as np
import xarray as xr
import matplotlib.path as mpath

def nice_boundary_path_for_maps(lon,lat):
    
    Path = mpath.Path
    path_data = [(Path.MOVETO, (lon.min(), lat.min()))]

    for lo in lon:
        path_data.append((Path.LINETO, (lo, lat.min())))
    
    path_data.append((Path.LINETO, (lon.max(), lat.max())))

    for lo in np.flip(lon):
        path_data.append((Path.LINETO, (lo, lat.max())))
    
    path_data.append((Path.CLOSEPOLY, (lon.min(), lat.min())))
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)

    return path

def define_grid(N):
    delta=[]
    for i in range(1,N):
        j = np.ceil(N/i)
        delta = np.append(delta,abs(j-i))
    
    return int(np.argmin(delta)+1), int(np.ceil(N/(np.argmin(delta)+1)))

def extract_semidiurnal_wave2(
    u: xr.DataArray,
    time_dim: str = "time",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    period_hours: float = 12.0,
    zonal_wavenumber: int = 2,
    include_both_directions: bool = True,
    time_in_hours: xr.DataArray | None = None,
) -> xr.Dataset:
    """
    Extrahiere die semidiurnale Wave-k-Komponente aus einem Feld u(time, lat, lon)
    mittels harmonischer Regression in Zeit und Länge.

    Parameter
    ---------
    u : xr.DataArray
        Eingabefeld mit Dimensionen (time, lat, lon), z.B. Zonalwind.
    time_dim, lat_dim, lon_dim : str
        Namen der Dimensionen.
    period_hours : float
        Periodendauer der Gezeit in Stunden. Für semidiurnal: 12.
    zonal_wavenumber : int
        Zonale Wellenzahl, für Wave-2 also 2.
    include_both_directions : bool
        Wenn True, fit auf beide Terme:
            cos(k*lon - omega*t), sin(k*lon - omega*t),
            cos(k*lon + omega*t), sin(k*lon + omega*t)
        Wenn False, nur:
            cos(k*lon - omega*t), sin(k*lon - omega*t)
    time_in_hours : xr.DataArray | None
        Optionale Zeitkoordinate in Stunden. Falls None, wird versucht, sie aus
        u[time_dim] zu bestimmen. Bei datetime64 wird die Zeit relativ zum ersten
        Zeitschritt in Stunden berechnet.

    Returns
    -------
    xr.Dataset
        Dataset mit rekonstruierter Tide und Fitparametern.
    """
    # --- Checks
    required_dims = {time_dim, lat_dim, lon_dim}
    missing = required_dims - set(u.dims)
    if missing:
        raise ValueError(f"u is missing required dimensions: {missing}")

    # In definierte Reihenfolge bringen
    u = u.transpose(time_dim, lat_dim, lon_dim)

    # Koordinaten holen
    time_coord = u[time_dim]
    lat = u[lat_dim]
    lon = u[lon_dim]

    nt = u.sizes[time_dim]
    nlat = u.sizes[lat_dim]
    nlon = u.sizes[lon_dim]

    # Längen in Radiant
    lon_rad = np.deg2rad(lon.values)

    # Zeit in Stunden bestimmen
    if time_in_hours is None:
        if np.issubdtype(time_coord.dtype, np.datetime64):
            t_hours = (
                (time_coord.values - time_coord.values[0]) / np.timedelta64(1, "h")
            ).astype(float)
        else:
            t_hours = np.asarray(time_coord.values, dtype=float)
    else:
        t_hours = np.asarray(time_in_hours.values, dtype=float)

    omega = 2.0 * np.pi / period_hours
    k = zonal_wavenumber

    # 2D-Gitter Zeit/Länge
    TT, LL = np.meshgrid(t_hours, lon_rad, indexing="ij")

    # Basisfunktionen
    Xm1 = np.cos(k * LL - omega * TT).reshape(-1)
    Xm2 = np.sin(k * LL - omega * TT).reshape(-1)
    X0 = np.ones_like(Xm1)

    if include_both_directions:
        Xp1 = np.cos(k * LL + omega * TT).reshape(-1)
        Xp2 = np.sin(k * LL + omega * TT).reshape(-1)
        X = np.column_stack([X0, Xm1, Xm2, Xp1, Xp2])
        ncoef = 5
    else:
        X = np.column_stack([X0, Xm1, Xm2])
        ncoef = 3

    # Ergebnisse vorbereiten
    tide = np.full((nt, nlat, nlon), np.nan, dtype=float)
    mean_field = np.full(nlat, np.nan, dtype=float)

    a_minus = np.full(nlat, np.nan, dtype=float)
    b_minus = np.full(nlat, np.nan, dtype=float)
    amp_minus = np.full(nlat, np.nan, dtype=float)
    phase_minus = np.full(nlat, np.nan, dtype=float)

    if include_both_directions:
        a_plus = np.full(nlat, np.nan, dtype=float)
        b_plus = np.full(nlat, np.nan, dtype=float)
        amp_plus = np.full(nlat, np.nan, dtype=float)
        phase_plus = np.full(nlat, np.nan, dtype=float)

    # Fit für jede Breite separat
    for i in range(nlat):
        y2d = u.isel({lat_dim: i}).values  # shape (time, lon)
        y = y2d.reshape(-1)

        valid = np.isfinite(y)
        if valid.sum() < ncoef:
            continue

        coef, *_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)

        if include_both_directions:
            c0, am, bm, ap, bp = coef
            mean_field[i] = c0

            a_minus[i] = am
            b_minus[i] = bm
            amp_minus[i] = np.sqrt(am**2 + bm**2)
            phase_minus[i] = np.arctan2(bm, am)

            a_plus[i] = ap
            b_plus[i] = bp
            amp_plus[i] = np.sqrt(ap**2 + bp**2)
            phase_plus[i] = np.arctan2(bp, ap)

            tide[:, i, :] = (
                am * np.cos(k * LL - omega * TT)
                + bm * np.sin(k * LL - omega * TT)
                + ap * np.cos(k * LL + omega * TT)
                + bp * np.sin(k * LL + omega * TT)
            )
        else:
            c0, am, bm = coef
            mean_field[i] = c0

            a_minus[i] = am
            b_minus[i] = bm
            amp_minus[i] = np.sqrt(am**2 + bm**2)
            phase_minus[i] = np.arctan2(bm, am)

            tide[:, i, :] = (
                am * np.cos(k * LL - omega * TT)
                + bm * np.sin(k * LL - omega * TT)
            )

    # Ausgabe als xarray.Dataset
    ds = xr.Dataset(
        data_vars={
            "tide": ((time_dim, lat_dim, lon_dim), tide),
            "mean_fit": ((lat_dim,), mean_field),
            "a_minus": ((lat_dim,), a_minus),
            "b_minus": ((lat_dim,), b_minus),
            "amplitude_minus": ((lat_dim,), amp_minus),
            "phase_minus": ((lat_dim,), phase_minus),
        },
        coords={
            time_dim: time_coord,
            lat_dim: lat,
            lon_dim: lon,
        },
        attrs={
            "description": "Extracted semidiurnal zonal wave component from harmonic regression",
            "period_hours": period_hours,
            "zonal_wavenumber": zonal_wavenumber,
            "include_both_directions": include_both_directions,
        },
    )

    if include_both_directions:
        ds["a_plus"] = ((lat_dim,), a_plus)
        ds["b_plus"] = ((lat_dim,), b_plus)
        ds["amplitude_plus"] = ((lat_dim,), amp_plus)
        ds["phase_plus"] = ((lat_dim,), phase_plus)

    # Attribute übernehmen, wenn vorhanden
    if hasattr(u, "name") and u.name is not None:
        ds["tide"].attrs["source_field"] = u.name
    if "units" in u.attrs:
        ds["tide"].attrs["units"] = u.attrs["units"]
        ds["mean_fit"].attrs["units"] = u.attrs["units"]
        ds["amplitude_minus"].attrs["units"] = u.attrs["units"]
        if include_both_directions:
            ds["amplitude_plus"].attrs["units"] = u.attrs["units"]

    ds["phase_minus"].attrs["units"] = "radian"
    if include_both_directions:
        ds["phase_plus"].attrs["units"] = "radian"

    return ds
