#!/usr/bin/env python3
import pathlib

import fsspec
import numcodecs
import numpy as np
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("time",):
            chunks = {
                "time": 2**18,
            }
        case ("time", "particle_size"):
            chunks = {
                "time": 2**18,
                "particle_size": 32,
            }
        case ("time", "raw_fall_velocity"):
            chunks = {
                "time": 2**18,
                "raw_fall_velocity": 32,
            }
        case ("time", *_):
            chunks = {
                "time": 2**18,
                "raw_fall_velocity": 32,
                "particle_size": 32,
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


if __name__ == "__main__":
    root = "ipfs://QmR5UvwZgpuQRfyKHqmirkYgfHDskPMgL49BhaS2ezgW1x"

    for f in fsspec.filesystem("ipfs").glob(f"{root}/*.nc"):
        ds = xr.open_dataset(fsspec.open_local(f"simplecache::ipfs://{f}"))

        ds.attrs["summary"] = "\n".join(
            line.strip() for line in ds.attrs["summary"].split("\n")
        )
        ds.attrs["project"] = ",".join(ds.attrs["project"])
        ds.attrs["featureType"] = "trajectory"

        ds.attrs["history"] = (
            "converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
        )

        ds.to_zarr(
            pathlib.Path(f).name.replace(".nc", ".zarr"),
            encoding=get_encoding(ds),
            mode="w",
            zarr_format=2,
        )
