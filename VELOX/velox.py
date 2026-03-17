#!/usr/bin/env python3
#SBATCH --partition=compute
#SBATCH --account=mh0066
#SBATCH --time=01:00:00
#SBATCH --time=01:00:00
#SBATCH --mem=0
import os
import datetime

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
        case ("time", "y", "x", *opt):
            chunks = {
                "time": 30,
                "y": 102,
                "x": 106,
                **{d: 1 for d in opt},
            }
        case ("y", "x"):
            chunks = {
                "y": sizes["y"],
                "x": sizes["x"],
            }
        case (any_time, "wavelength"):
            chunks = {
                any_time: 2**16,
                "wavelength": 4,
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
            "chunks": get_chunks(var, da.sizes),
        }
        for var, da in dataset.variables.items()
    }


def round_datetime(ds):
    return ds.assign_coords(
        time=(ds.time.values + np.timedelta64(500, "ms")).astype("datetime64[s]")
    )


def open_dataset(date="20200124"):
    print(date, flush=True)
    data_root = f"/work/mh0010/ORCESTRA/raw/HALO/velox/HALO-{date}a"
    wavelengths = [8650, 10740, 11660, 12000]

    datasets = []
    for wl in wavelengths:
        ds = xr.open_mfdataset(
            f"{data_root}/PERCUSION_HALO_VELOX_BT?_{wl}nm_{date}a.nc",
            chunks={"x": -1, "y": -1, "time": 30, "time_sim": -1},
        ).pipe(round_datetime)

        BT_2D = ds.BT_2D.expand_dims("wavelength", axis=-1).assign_attrs(
            long_name="Two-dimensional brightness temperature of VELOX at flight altitude",
            standard_name="brightness_temperature",
        )

        BT_center = ds.BT_center.expand_dims("wavelength", axis=-1).assign_attrs(
            standard_name="brightness_temperature",
        )

        BT_sim = ds.BT_sim.expand_dims("wavelength", axis=-1).assign_attrs(
            long_name="Simulated brightness temperature",
            standard_name="brightness_temperature",
        )

        ds = ds.assign(
            BT_2D=BT_2D,
            BT_center=BT_center,
            BT_sim=BT_sim,
            vaa=ds.vaa.assign_attrs(standard_name="sensor_azimuth_angle"),
            vza=ds.vza.assign_attrs(standard_name="sensor_zenith_angle"),
            lat=ds.lat.assign_attrs(standard_name="latitude", units="degrees_north"),
            lon=ds.lon.assign_attrs(standard_name="longitude", units="degrees_east"),
        ).assign_coords(
            time=ds.time.assign_attrs(standard_name="time"),
            wavelength=(
                ("wavelength",),
                np.array([wl]),
                {"standard_name": "radiation_wavelength", "units": "nm"},
            ),
        )

        datasets.append(ds)

    ds_merged = xr.concat(
        datasets,
        dim="wavelength",
        data_vars="minimal",
        compat="override",
        coords="minimal",
        join="outer",
    )

    ds_broadband = xr.open_dataset(
        f"{data_root}/PERCUSION_HALO_VELOX_BT1_7700-12000nm_{date}a.nc",
        chunks={"x": -1, "y": -1, "time": 30},
    ).pipe(round_datetime)

    ds_merged = ds_merged.assign(
        BT_2D_broadband=ds_broadband.BT_2D.assign_attrs(
            standard_name="brightness_temperature"
        ),
        BT_center_broadband=ds_broadband.BT_center.assign_attrs(
            standard_name="brightness_temperature"
        ),
        BT_sim_broadband=ds_broadband.BT_sim.assign_attrs(
            standard_name="brightness_temperature",
            long_name="Simulated brightness temperature",
        ),
    ).chunk(time=30, time_sim=-1)

    for var, da in ds_merged.variables.items():
        if len(da.dims) == 1:
            ds_merged = ds_merged.assign({var: da.chunk({da.dims[0]: -1})})
        elif var in ("BT_center", "BT_sim", "BT_center_broadband", "BT_sim_broadband"):
            ds_merged = ds_merged.assign({var: da.chunk(time=-1, wavelength=4)})

    ds_merged = ds_merged.assign_attrs(
        title="Two-dimensional brightness temperature (1 Hz) derived from VELOX during the ORCESTRA field campaign",
        creator_name="Sophie Rosenburg, Michael Schäfer, Anna E. Luebke, Kevin Wolf, Patrizia Schoch, André Ehrlich, Manfred Wendisch",
        creator_email="sophie.rosenburg@uni-leipzig.de, michael.schaefer@uni-leipzig.de, anna.luebke@uni-leipzig.de, kevin.wolf@uni-leipzig.de, patrizia.schoch@uni-leipzig.de, a.ehrlich@uni-leipzig.de, m.wendisch@uni-leipzig.de",
        featureType="timeSeries",
        description=open("description.txt").read(),
        keywords="PERCUSION, HALO, VELOX, thermal-infrared, brightness temperature",
        reference="Schäfer, M. et al: VELOX – A new thermal infrared imager for airborne remote sensing of cloud and surface properties, Atmos. Meas. Tech., 15, 1491-1509, https://doi.org/10.5194/amt-15-1491-2022, 2022.",
        license="CC-BY-4.0",
        history=f"{ds.created_on}: Created, {datetime.date.today()}: Merged datasets over campaign period and converted to Zarr",
    )
    ds_merged.attrs.pop("created_on")
    ds_merged.attrs.pop("instrument_reference")
    ds_merged.attrs.pop("recommended_usage")
    ds_merged.attrs.pop("authors")
    ds_merged.attrs.pop("contact")

    return ds_merged


def process_flights():
    flight_dates = (
        "20240811",
        "20240813",
        "20240816",
        "20240818",
        "20240821",
        "20240822",
        "20240825",
        "20240827",
        "20240829",
        "20240831",
        "20240903",
        "20240906",
        "20240907",
        "20240909",
        "20240912",
        "20240914",
        "20240916",
        "20240919",
        "20240921",
        "20240923",
        "20240924",
        "20240926",
        "20240928",
        "20241105",
        "20241107",
        "20241110",
        "20241112",
        "20241114",
        "20241116",
    )

    date = flight_dates[int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))]
    ds = open_dataset(date)
    ds.to_zarr(
        f"/scratch/m/m300575/HALO_VELOX_BT_Flight_{date}.zarr",
        zarr_format=2,
        mode="w",
        encoding=get_encoding(ds),
    )


def concat_stores():
    ds_full = xr.open_mfdataset(
        "zarr/HALO_VELOX_BT_Flight_*.zarr",
        data_vars="all",
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
    )

    # store = "swift://swift.dkrz.de/dkrz_948e7d4bbfbb445fbff5315fc433e36a/ORCESTRA/HALO_VELOX_BT.zarr"
    store = "/scratch/m/m300575/HALO_VELOX_BT.zarr"

    ds_full.chunk(time=-1).to_zarr(
        store,
        mode="w",
        zarr_format=2,
        encoding=get_encoding(ds_full),
        compute=False,
        storage_options={"get_client": get_client},
    )

    ds_full[["alt", "lat", "lon", "vaa", "vza"]].chunk(time=-1).to_zarr(
        store,
        mode="a",
        storage_options={"get_client": get_client},
    )

    ds_full[["BT", "BT_broadband"]].chunk(time=30).to_zarr(
        store,
        mode="a",
        storage_options={"get_client": get_client},
    )


if __name__ == "__main__":
    process_flights()
