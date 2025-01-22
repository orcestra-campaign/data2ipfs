#!/usr/bin/env python
# coding: utf-8

import bz2
import glob
import xarray as xr
import netCDF4 as nc4
import numcodecs
import numpy as np
import pandas as pd
import re
from datetime import datetime
from zoneinfo import ZoneInfo


# ## Read PARCIVEL
# define 1dim fields to read
fields_single = [
    ("rain_intensity", "<f4", {"long_name": "Rain Intensity", "units": "mm h-1"}),
    ("rain_amount", "<f4", {"long_name": "Rain amount accumulated", "units": "mm"}),
    ("weather_code", "<i2", {"long_name": "Weather Code"}),
    ("radar_reflectivity", "<f4", {"long_name": "Radar Reflectivity", "units": "0.1 lg(re 1 mm)"}),
    ("mor_visibility", "<i2", {"long_name": "MOR visibility", "units": "m"}),
    ("sample_interval", "<i2", {"long_name": "Sample Interval", "units": "s"}),
    ("signal_amplitude", "<i4", {"long_name": "Signal amplitude"}),
    ("number_of_particles", "<i4", {"long_name": "Number of particles", "units": "1"}),
    ("temperature_housing", "<i2", {"long_name": "Temperatur housing", "units": "degC"}),
    ("sensor_serial_number", "<i4", {"long_name": "Sensor serial number"}),
    ("firmware_iop", "|S6", {"long_name": "Firmware IOP"}),
    ("heating_current", "<f4", {"long_name": "Heating current", "units": "A"}),
    ("supply_voltage", "<f4", {"long_name": "Power supply voltage", "units": "V"}),
    ("sensor_status", "<i2", {"long_name": "Sensor status"}),
    ("station_name", "|S10", {"long_name": "Station name"}),
    ("rain_amount_abs", "<f4", {"long_name": "Rain amount abs.", "units": "mm"}),
    ("error_code", "<i2", {"long_name": "Error Code"}),
]
# define arrays to read
fields_multi = [
    ("field_n_ved", "<f4", (32,), ("d",), {"long_name": "mean volume equivalent diameter per class", "units": "0.1 lg(re 1 m3 mm)"}),
    ("field_v", "<f4", (32,), ("v",), {"long_name": "mean velocity per class", "units": "mm s-1"}),
    ("raw", "<i2", (32, 32), ("v", "d"), {"long_name": "particle count per class"}),
]
#define boundaries
d_bounds = np.concatenate([
    np.arange(0, 10) * 0.125,
    np.arange(0, 5) * 0.25 + 1.25,
    np.arange(0, 5) * 0.5 + 2.5,
    np.arange(0, 5) * 1 + 5,
    np.arange(0, 5) * 2 + 10,
    np.arange(0, 3) * 3 + 20,
])
d_bounds = np.stack([d_bounds[:-1], d_bounds[1:]], axis=-1).astype("<f4")
d_centers = d_bounds.mean(axis=-1)

v_bounds = np.concatenate([
    np.arange(0, 10) * 0.1,
    np.arange(0, 5) * 0.2 + 1,
    np.arange(0, 5) * 0.4 + 2,
    np.arange(0, 5) * 0.8 + 4,
    np.arange(0, 5) * 1.6 + 8,
    np.arange(0, 3) * 3.2 + 16,
])
v_bounds = np.stack([v_bounds[:-1], v_bounds[1:]], axis=-1).astype("<f4")
v_centers = v_bounds.mean(axis=-1)


def is_empty_csv(filename):
    with bz2.open(filename, "r") as fp:
        for i, _ in enumerate(fp):
            if i > 0:
                return False

    return True


def fix_ds(ds):
    ofs = 2
    variables = {}
    for name, dtype, attrs in fields_single:
        variables[name] = ds[ofs].assign_attrs(attrs).astype(dtype)
        ofs += 1

    for name, dtype, shape, dims, attrs in fields_multi:
        n = np.prod(shape)
        var = np.stack([ds[i].values for i in range(ofs, ofs+n)], axis=-1).reshape(-1, *shape)
        variables[name] = xr.DataArray(var, dims=tuple(ds.dims) + dims, attrs=attrs).astype(dtype)
        ofs += n
    
    return xr.Dataset({
        **variables,
        "d_bounds": (("d", "bnd"), d_bounds, {"long_name": "bounds of particle diameter bins", "units": "mm"}),
        "v_bounds": (("v", "bnd"), v_bounds, {"long_name": "bounds of particle velocity bins", "units": "m s-1"}),
    }, coords={
        "d": ("d", d_centers, {"long_name": "nominal centers of particle diameter bins", "units": "mm", "bounds": "d_bounds"}),
        "v": ("v", v_centers, {"long_name": "nominal centers of particle velocity bins", "units": "m s-1", "bounds": "v_bounds"}),
    })


def read_parsivel(filename):
    # if is_empty_csv(filename):
    #     return xr.Dataset()


    df = pd.read_csv(filename,
                     delimiter=";",
                     skiprows=1,
                     header=None,
                     dtype={0: str, 1: str},
                     on_bad_lines="warn",
                     names=list(range(1108))  # specifying the known numbers of columns helps detecting broken lines
                    )

    df.pop(1107)  # empty column at end
    df = df.dropna()  # in some cases, broken lines still show up, but contain NaN values

    dtcols = df.pop(0) + ';' + df.pop(1)
    df['time'] = pd.to_datetime(dtcols, format="mixed", dayfirst=True)

    df = df.set_index('time')

    assert df.index.is_monotonic_increasing

    return fix_ds(df.to_xarray())


def get_chunks(dimensions):
    match dimensions:
        case ("time",):
            chunks = {
                "time": 2**18,
            }
        case ("time", "d"):
            chunks = {
                "time": 2**13,
                "d": 32,
            }
        case ("time", "v"):
            chunks = {
                "time": 2**13,
                "v": 32,
            }
        case ("time", "v", "d"):
            chunks = {
                "time": 2**8,
                "v": 32,
                "d": 32,
            }
        case ("v", "bnd"):
            chunks = {
                "v": 32,
                "bnd": 2,
            }
        case ("d", "bnd"):
            chunks = {
                "d": 32,
                "bnd": 2,
            }

    return tuple((chunks[d] for d in dimensions))


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd", shuffle=1, clevel=6)

    return {
        var: {
            "chunks": get_chunks(dataset[var].dims),
            "compressor": None if isinstance(dataset[var].dtype, np.dtypes.StrDType) else codec,
        }
        for var in dataset.variables
        if var not in dataset.dims
    }


if __name__ == "__main__":
    for instrument in ("Parsivel_1", "Parsivel_2"):
        ds = read_parsivel(f"/scratch/m/m300575/tmp/{instrument}/{instrument}.csv")

        # Add constant fields as global attributes
        ds.attrs["firmware_iop"] = ds.firmware_iop.values[0].decode()
        ds.attrs["station_name"] = ds.station_name.values[0].decode()
        ds.attrs["sensor_serial_numer"] = ds.sensor_serial_number[0].values.item()
        ds = ds.drop_vars(["firmware_iop", "station_name", "sensor_serial_number"])

        ds.attrs["title"] = f"{instrument.replace('_', ' ')} disdrometer measurements during METEOR cruise M203"
        ds.attrs["creator_name"] = "Friedhelm Jansen"
        ds.attrs["creator_email"] = "friedhelm.jansen@mpimet.mpg.de"
        ds.attrs["project"] = "ORCESTRA, BOW-TIE"
        ds.attrs["platform"] = "RV METEOR"
        ds.attrs["source"] = "OTT Parsivel2 laser disdrometer"
        ds.attrs["license"] = "CC-BY-4.0"

        now = datetime.now().astimezone(ZoneInfo("UTC")).strftime(r"%Y-%m-%dT%H:%M:%SZ")
        ds.attrs["history"] = (
            f"{now} converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
        )

        ds.to_zarr(f"{instrument}.zarr", encoding=get_encoding(ds), mode="w", zarr_format=2)
