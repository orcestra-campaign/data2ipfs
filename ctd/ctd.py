#!/usr/bin/env python3
import glob
from datetime import datetime
from zoneinfo import ZoneInfo

import fsspec
import numcodecs
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("SOUNDING",):
            chunks = {
                "SOUNDING": -1,
            }
        case ("SOUNDING", "PRES"):
            chunks = {
                "SOUNDING": -1,
                "PRES": 2240,
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


def open_dataset(filename):
    ds = xr.open_dataset(filename, decode_cf=False)
    ds.FULL_TIME.values -= 712224
    ds = xr.decode_cf(ds)

    ds = ds.swap_dims(
        {"TIME": "SOUNDING", "LATITUDE": "SOUNDING", "LONGITUDE": "SOUNDING"}
    )

    return ds


def main():
    root = "ipns://latest.orcestra-campaign.org"
    datasets = [
        open_dataset(fsspec.open_local(f"simplecache::ipns://{f}"))
        for f in sorted(
            fsspec.filesystem("ipns").glob(f"{root}/raw/METEOR/CTD/nc/met_203_1_ctd_*.nc")
        )
    ]
    ds = xr.concat(datasets, dim="SOUNDING")
    ds = ds.assign_coords(SOUNDING=range(1, ds.sizes["SOUNDING"] + 1))

    now = datetime.now().astimezone(ZoneInfo("UTC")).strftime(r"%Y-%m-%dT%H:%M:%SZ")

    ds.attrs["title"] = "GEOMAR PO CTD data for METEOR cruise M203"
    ds.attrs["history"] += (
        f"; {now} converted to zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    )
    ds.attrs["publisher_email"] = "lukas.kluft@mpimet.mpg.de"
    ds.attrs["publisher_name"] = "Lukas Kluft"
    ds.attrs["publisher_url"] = "https://orcid.org/0000-0002-6533-3928"

    ds.to_zarr("CTD.zarr", encoding=get_encoding(ds), mode="w")


if __name__ == "__main__":
    main()
