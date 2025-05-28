#!/usr/bin/env python3
import pathlib

import fsspec
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("time",):
            chunks = {
                "time": 2**18,
            }

    return tuple((chunks[d] for d in dimensions))


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd", shuffle=1, clevel=6)

    return {
        var: {
            "chunks": get_chunks(dataset[var].dims),
            "compressor": None
            if isinstance(dataset[var].dtype, np.dtypes.StrDType)
            else codec,
        }
        for var in dataset.variables
        if var not in dataset.dims
    }


# +
def _main():
    # Parse CSV log
    root = "ipfs://QmdxMqRNRKrp9sWPCumSRMoaBKKAMTASSymYVUDswokH73"
    df = pd.read_csv(
        fsspec.open_local(f"simplecache::{root}"),
        sep=";",
    )

    # Parse date/time information and convert to Xarray
    dtcols = df.pop("Timestamp")
    df["time"] = pd.to_datetime(dtcols, format="mixed", dayfirst=True)
    ds = df.set_index("time").to_xarray()

    # Add height information
    ds = ds.assign(height=((), 35.0, {"long_name": "height", "units": "m"}))

    # Tro2 dauerhaft deaktiviert
    ds = ds.drop_vars("Tro2")

    # Convert unparsed values
    ds = ds.assign(
        RR_PWD22=(("time",), np.array([float(v) for v in ds.RR_PWD22.values]))
    )

    # Attach variable names
    variables = (
        ("Lat", "latitude", "degrees_north"),
        ("Long", "longitude", "degrees_east"),
        ("RR_SRM", "rain rate", "mm/h"),
        ("Dauer", "duration", "s"),
        ("Tro1", "droplets (downward)", "1"),
        ("Trs", "droplets (lateral)", "1"),
        ("FF", "wind speed (relative)", "m/s"),
        ("DD", "wind direction", "degrees"),
        ("TT", "temperatur", "degC"),
        ("RH", "relative humidity", "%"),
        ("VVV", "visibility", "m"),
        ("RR_PWD22", "rain rate", "mm/h"),
    )

    for var, long_name, units in variables:
        ds[var].attrs = {
            "long_name": long_name,
            "units": units,
            "coordinates": "time height",
        }

    # Add metadata
    ds.attrs["title"] = "Rain gauge measurements during METEOR cruise M203"
    ds.attrs["summary"] = (
        "The rain gauge has an upper and a lateral collecting surface on which the precipitation is converted into digital pulses via droplet formers. The pulses are recorded by the data logger as counting pulses via two digital inputs and transmitted to the meteorological computer via the RS422-interface.\n\n"
        "The precipitation detector serves as a signalling device for determining the start and end of precipitation. The precipitation is recorded by a light barrier system and triggers a switching signal, which is recorded via a digital status input of the data logger. The precipitation duration is determined in the METCO by integrating the status signal over time.\n\n"
        "To determine the precipitation rate, the PWD22 visibility gauge is also used, which determines the type of precipitation by determining the water content of precipitation using a capacitive measurement (Vaisala RAINCAP® sensor element) and combining this information with the results from the optical forward scattering measurement. To increase sensitivity, the system is equipped with two Vaisala RAINCAP® sensors, allowing even the slightest precipitation, such as light drizzle, to be detected. By using advanced algorithms, it is possible to determine the type, amount and intensity of precipitation with high accuracy.\n\n"
        "The sensitivity of the precipitation detection is 0.05 mm/h or less."
    )
    ds.attrs["keywords"] = "precipitation amount, precipitation gauge, precipitation rate"

    ds.attrs["creator_name"] = "Martin Stelzner, Daniel Klocke"
    ds.attrs["creator_email"] = "Martin.Stelzner@dwd.de, daniel.klocke@mpimet.mpg.de"

    ds.attrs["project"] = "ORCESTRA, BOW-TIE"
    ds.attrs["platform"] = "RV METEOR"
    ds.attrs["featureType"] = "trajectory"

    ds.attrs["license"] = "CC-BY-4.0"
    ds.attrs["history"] = "Converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"

    # Store to Zarr
    ds.to_zarr(
        "M203_Niederschlag_Stand_240923-2227.zarr",
        encoding=get_encoding(ds),
        mode="w",
        zarr_format=2,
    )


if __name__ == "__main__":
    _main()
