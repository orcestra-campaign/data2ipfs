import argparse

import cartopy.crs as ccrs
import numcodecs
import numpy as np
import xarray as xr


def attach_xy_coordinates(ds):
    # Attach `coordinates` and `grid_mapping` attributes to initial variables
    for var in ds.variables:
        if "lat" in ds[var].dims and "lon" in ds[var].dims:
            ds = ds.assign(
                {
                    var: ds[var].assign_attrs(
                        grid_mapping="crs",
                        coordinates="lat lon",
                    )
                }
            )

    # Compute x and y coordinates in geostationary projection
    seviri_proj = ccrs.Geostationary()

    valid_lons = ds.lons.where(ds.lons != -999)
    valid_lats = ds.lats.where(ds.lats != -999)

    ds = ds.assign(lons=valid_lons, lats=valid_lats)

    xyz = seviri_proj.transform_points(
        src_crs=ccrs.Geodetic(),
        x=valid_lons.values,
        y=valid_lats.values,
        trap=True,
    )

    # Appxorimate 1d coordinates
    x = np.nanmean(xyz[..., 0], axis=0)
    y = np.nanmean(xyz[..., 1], axis=1)

    x = xr.DataArray(
        data=x,
        dims=("x",),
        attrs={
            "units": "m",
            "long_name": "x coordinate of projection",
            "standard_name": "projection_x_coordinate",
        },
    )

    y = xr.DataArray(
        data=y,
        dims=("y",),
        attrs={
            "units": "m",
            "long_name": "y coordinate of projection",
            "standard_name": "projection_y_coordinate",
        },
    )

    # Swap lat/lon coordinates with x/y (and remove NaN stripe)
    ds = ds.swap_dims(lon="x", lat="y").assign_coords(x=x, y=y).isel(x=~np.isnan(x))

    # Attach global CRS attribute
    crs = xr.DataArray(
        name="crs",
        attrs={
            "grid_mapping_name": "geostationary",
        },
    )
    ds = ds.assign_coords(crs=crs)

    return ds


def get_chunks(sizes):
    match tuple(sizes.keys()):
        case ("channel", "time", "y", "x"):
            chunks = {
                "channel": 1,
                "time": 8,
                "y": 280,
                "x": 210,
            }
        case ("y", "x"):
            chunks = {
                "y": sizes["y"],
                "x": sizes["x"],
            }
        case (single_dim,):
            chunks = {
                single_dim: sizes[single_dim],
            }
        case _:
            chunks = {}

    return tuple((chunks[d] for d in sizes))


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    compressor = numcodecs.Blosc("zstd", clevel=6)

    return {
        var: {
            "compressor": compressor,
            "chunks": get_chunks(dataset[var].sizes),
        }
        for var in dataset.variables
    }


def main(infile, outfile):
    # Open dataset and attach geostationaty xy-coordinates
    ds = xr.open_dataset(
        infile,
        engine="zarr",
        chunks={},
    ).pipe(attach_xy_coordinates)

    # Reduce precision for omega standard deviation
    ds = ds.assign(err_omega=ds.err_omega.astype("<f4"))

    # Fix attribute conventions
    ds = ds.assign_attrs(
        summary=ds.attrs.pop("description"),
        references=f'"{ds.references}"',
        platform="MSG",
    )

    # Write Zarr metadata
    ds.chunk(channel=-1, time=-1, x=-1, y=-1).to_zarr(
        outfile,
        encoding=get_encoding(ds),
        mode="w",
        zarr_format=2,
        compute=False,
    )

    # Write data chunks (per variable)
    for varname in ds.variables:
        print(varname)
        ds[[varname]].load().to_zarr(outfile, mode="a")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--infile",
        default="/work/bb1153/b381959/ORCESTRA/omega_ORCESTRA_new.zarr",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="/scratch/m/m300575/omega_ORCESTRA.zarr",
    )
    args = parser.parse_args()

    main(args.infile, args.outfile)
