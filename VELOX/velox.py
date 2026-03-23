#!/usr/bin/env python3
#SBATCH --partition=compute
#SBATCH --account=mh0066
#SBATCH --time=04:00:00
#SBATCH --mem=0
import os
import datetime
import pathlib

import numcodecs
import numpy as np
import xarray as xr


async def get_client(**kwargs):
    import aiohttp
    import aiohttp_retry

    retry_options = aiohttp_retry.ExponentialRetry(
        attempts=3, exceptions={OSError, aiohttp.ServerDisconnectedError}
    )
    retry_client = aiohttp_retry.RetryClient(
        raise_for_status=False, retry_options=retry_options
    )
    return retry_client


def get_chunks(var, sizes):
    match tuple(sizes.keys()):
        case ("time", "y", "x"):
            chunks = {
                "time": 30,
                "y": 102,
                "x": 106,
            }
        case ("y", "x"):
            chunks = {
                "y": sizes["y"],
                "x": sizes["x"],
            }
        case ("time",) | ("time_sim",):
            chunks = {
                list(sizes)[0]: 2**18,
            }
        case (single_dim,):
            chunks = {
                single_dim: sizes[single_dim],
            }
        case _:
            chunks = {}

    print(var, sizes, chunks)
    return tuple((chunks[d] for d in sizes))


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    compressor = numcodecs.Blosc("zstd", clevel=6)

    return {
        var: {
            "compressor": compressor,
            "chunks": get_chunks(var, da.sizes),
        }
        for var, da in dataset.variables.items()
    }


def filter_by_dim(ds, dim):
    return ds[[v for v, a in ds.variables.items() if dim in a.dims]]


def concat_along_dim(datasets, dim):
    return xr.concat(
        (filter_by_dim(d, dim) for d in datasets),
        dim=dim,
        compat="override",
        coords="minimal",
        combine_attrs="drop_conflicts",
    )


def open_dataset(wavelength):
    print(wavelength, flush=True)
    data_root = pathlib.Path("/work/mh0010/ORCESTRA/raw/HALO/velox/")

    datasets = []
    for ncfile in data_root.glob(f"*/*{wavelength}*.nc"):
        ds = xr.open_mfdataset(
            ncfile,
            chunks={"x": -1, "y": -1, "time": 30, "time_sim": -1},
        )

        ds = (
            ds.assign(
                BT_2D=ds.BT_2D.assign_attrs(
                    long_name="Two-dimensional brightness temperature measured by VELOX at flight altitude",
                    standard_name="brightness_temperature",
                ),
                BT_center=ds.BT_center.assign_attrs(
                    standard_name="brightness_temperature",
                ),
                BT_sim=ds.BT_sim.assign_attrs(
                    long_name="Simulated brightness temperature",
                    standard_name="brightness_temperature",
                ),
                vaa=ds.vaa.assign_attrs(standard_name="sensor_azimuth_angle"),
                vza=ds.vza.assign_attrs(standard_name="sensor_zenith_angle"),
                lat=ds.lat.assign_attrs(
                    standard_name="latitude", units="degrees_north"
                ),
                lon=ds.lon.assign_attrs(
                    standard_name="longitude", units="degrees_east"
                ),
            )
            .assign_coords(
                time=ds.time.assign_attrs(standard_name="time"),
            )
            .set_coords(("alt", "lat", "lon"))
        )

        datasets.append(ds)

    ds_merged = xr.merge(
        (
            concat_along_dim(datasets, dim="time"),
            concat_along_dim(datasets, dim="time_sim"),
            datasets[0][["vaa", "vza"]],
        )
    ).chunk(time=30, time_sim=-1)

    for var, da in ds_merged.variables.items():
        if len(da.dims) == 1:
            ds_merged = ds_merged.assign({var: da.chunk({da.dims[0]: -1})})
        elif var in ("BT_center", "BT_sim", "BT_center_broadband", "BT_sim_broadband"):
            ds_merged = ds_merged.assign({var: da.chunk(time=-1)})

    wavelength_repr = "–".join([f"{float(wl) / 1e3} µm" for wl in wavelength.split("-")])
    ds_merged = ds_merged.assign_attrs(
        title=f"Two-dimensional brightness temperature ({wavelength_repr}, 1 Hz) derived from VELOX during PERCUSION",
        creator_name="Sophie Rosenburg, Michael Schäfer, Anna E. Luebke, Kevin Wolf, Patrizia Schoch, André Ehrlich, Manfred Wendisch",
        creator_email="sophie.rosenburg@uni-leipzig.de, michael.schaefer@uni-leipzig.de, anna.luebke@uni-leipzig.de, kevin.wolf@uni-leipzig.de, patrizia.schoch@uni-leipzig.de, a.ehrlich@uni-leipzig.de, m.wendisch@uni-leipzig.de",
        featureType="trajectory",
        summary=open("description.txt").read(),
        keywords="PERCUSION, HALO, VELOX, thermal-infrared, brightness temperature",
        references="'Schäfer, M. et al: VELOX – A new thermal infrared imager for airborne remote sensing of cloud and surface properties, Atmos. Meas. Tech., 15, 1491-1509, https://doi.org/10.5194/amt-15-1491-2022, 2022.'",
        license="CC-BY-4.0",
        history=f"{datetime.date.today()}: Merged datasets over campaign period and converted to Zarr",
    )
    ds_merged.attrs.pop("instrument_reference")
    ds_merged.attrs.pop("recommended_usage")
    ds_merged.attrs.pop("authors")
    ds_merged.attrs.pop("contact")

    return ds_merged


def process_flights():
    wavelengths = ["8650", "10740", "11660", "12000", "7700-12000"]
    wavelength = wavelengths[int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))]
    ds = open_dataset(wavelength)

    ds.to_zarr(
        f"/scratch/m/m300575/HALO_VELOX_BT_{wavelength}nm.zarr",
        zarr_format=2,
        mode="w",
        encoding=get_encoding(ds),
        compute=True,
    )


if __name__ == "__main__":
    process_flights()
