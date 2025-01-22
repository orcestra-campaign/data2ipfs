#!/usr/bin/env python3
from datetime import datetime
from zoneinfo import ZoneInfo

import fsspec
import numcodecs
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
    }


def main():
    root = "ipns://latest.orcestra-campaign.org"
    ds = xr.open_dataset(
        fsspec.open_local(
            f"simplecache::{root}/raw/METEOR/GNSS_IWV/METEOR_GNSS_IWV_20240813_000030_20240923_195300.nc"
        ),
        chunks={},
    )

    ds.attrs["title"] = "IWV data from GNSS antenna on R/V METEOR"
    ds.attrs["creator_name"] = "Pierre Bosser"
    ds.attrs["creator_email"] = "pierre.bosser@ensta-bretagne.fr"
    ds.attrs["summary"] = ds.attrs.pop("methodology")
    ds.attrs["project"] = "ORCESTRA, BOW-TIE"
    ds.attrs["platform"] = "RV METEOR"
    ds.attrs["source"] = "GNNS antenna"
    ds.attrs["license"] = "CC-BY-4.0"
    ds.attrs["references"] = '"Bosser et al. 2021, ESSD"'

    now = datetime.now().astimezone(ZoneInfo("UTC")).strftime(r"%Y-%m-%dT%H:%M:%SZ")
    ds.attrs["history"] += (
        f"; {now}: converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    )

    ds.to_zarr("METEOR_GNSS_IWV.zarr", mode="w", encoding=get_encoding(ds))


if __name__ == "__main__":
    main()
