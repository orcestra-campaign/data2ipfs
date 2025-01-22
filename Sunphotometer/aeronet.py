#!/usr/bin/env python3
import pathlib
from datetime import datetime

import fsspec
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr


def get_chunks(dimensions):
    chunks = {
        "time": -1,
    }

    return tuple((chunks[d] for d in dimensions))


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd", shuffle=1, clevel=6)

    return {
        var: {
            "chunks": get_chunks(dataset[var].dims),
            "compressor": codec,
        }
        for var in dataset.variables
        if var not in dataset.dims
    }


def open_dataset(csvfile):
    ds = pd.read_csv(csvfile, skiprows=4).to_xarray()
    time = [
        np.datetime64(datetime.strptime(d + t, "%d:%m:%Y%H:%M:%S"), "ns")
        for d, t in zip(ds["Date(dd:mm:yyyy)"].data, ds["Time(hh:mm:ss)"].data)
    ]

    ds = (
        ds.swap_dims(index="time")
        .assign_coords(time=time)
        .drop_vars(["index", "Date(dd:mm:yyyy)", "Time(hh:mm:ss)"])
    )

    for var in ds.variables:
        if var not in ds.dims:
            ds[var].attrs["long_name"] = var
            ds = ds.rename({var: var.lower().replace(" ", "_")})

    ds.attrs = {
        "title": "2024 RV Meteor Cruise",
        "source": "AERONET Maritime Aerosol Network",
        "PI": "Elena Lind, Pawan Gupta and Daniel Klocke",
        "email": "elena.lind@nasa.gov, pawan.gupta@nasa.gov and daniel.klocke@mpimet.mpg.de",
        "references": "https://aeronet.gsfc.nasa.gov/new_web/cruises_v3/Meteor_24_0.html",
        "history": "Converted to Zarr by Lukas Kluft",
    }

    return ds


def main():
    protocol = "ipns"
    root = "ipns://latest.orcestra-campaign.org/raw/METEOR/sunphotometer"

    fs = fsspec.filesystem(protocol)
    for subdir, pattern in zip(("AOD", "SDA"), ("Meteor_24_0_*.lev??", "Meteor_24_0_*.ONEILL_??")):
        for csvfile in fs.glob(f"{root}/{subdir}/{pattern}"):
            csvfile = pathlib.Path(csvfile)

            ds = open_dataset(fsspec.open_local(f"simplecache::{protocol}://{csvfile}"))

            ds.to_zarr(
                csvfile.with_suffix(csvfile.suffix + ".zarr").name,
                mode="w",
                encoding=get_encoding(ds),
            )


if __name__ == "__main__":
    main()
