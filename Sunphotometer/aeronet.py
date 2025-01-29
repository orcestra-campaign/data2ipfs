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
        "creator_name": "Elena Lind, Pawan Gupta, Daniel Klocke",
        "creator_email": "elena.lind@nasa.gov, pawan.gupta@nasa.gov, daniel.klocke@mpimet.mpg.de",
        "project": "ORCESTRA, BOW-TIE, AERONET",
        "platform": "RV METEOR",
        "references": "https://aeronet.gsfc.nasa.gov/new_web/cruises_v3/Meteor_24_0.html",
        "license": "CC-BY-4.0",
        "history": "Converted to Zarr by Lukas Kluft",
        "data_usage": (
            "The public domain data you are about to download are contributed by the Maritime Aerosol Network (MAN), "
            "a component of the International AERONET Federation. Each cruise has Principal Investigators (PIs) "
            "responsible for instrument deployment, maintenance and data collection. The PIs have priority use of the "
            "data collected during the ship cruise. The PIs are entitled to be notified when using their cruise data. "
            "PIs contact information are available on data charts and in downloaded data files for each cruise."
        ),
        "featureType": "trajectory",
    }

    return ds


def main():
    protocol = "ipns"
    root = "ipns://latest.orcestra-campaign.org/raw/METEOR/sunphotometer"

    fs = fsspec.filesystem(protocol)
    for subdir, pattern in zip(
        ("AOD", "SDA"), ("Meteor_24_0_*.lev??", "Meteor_24_0_*.ONEILL_??")
    ):
        for csvfile in fs.glob(f"{root}/{subdir}/{pattern}"):
            ds = open_dataset(fsspec.open_local(f"simplecache::{protocol}://{csvfile}"))

            # Construct dataset title
            if "10" in csvfile:
                level = "Level 1.0"
                ds.attrs["source"] = (
                    f"{level} Maritime Aerosol Network (MAN) Measurements: These data are not screened and may not have final calibration applied"
                )
            elif "15" in csvfile:
                level = "Level 1.5"
                ds.attrs["source"] = (
                    f"{level} Maritime Aerosol Network (MAN) Measurements: These data are screened for clouds and pointing errors but may not have final calibration applied"
                )
            elif "20" in csvfile:
                level = "Level 2.0"
                ds.attrs["source"] = (
                    f"{level} Maritime Aerosol Network (MAN) Measurements: These data are screened for clouds and pointing errors, have final calibration applied, and are manually inspected"
                )

            retrieval = ", SDA Retrieval" if "ONEILL" in csvfile else ""

            ds.attrs["title"] = (
                f"Sunphotometer (Microtops) measurements during METEOR cruise M203 ({level}{retrieval})"
            )

            csvfile = pathlib.Path(csvfile)
            ds.to_zarr(
                csvfile.with_suffix(csvfile.suffix + ".zarr").name,
                mode="w",
                encoding=get_encoding(ds),
            )


if __name__ == "__main__":
    main()
