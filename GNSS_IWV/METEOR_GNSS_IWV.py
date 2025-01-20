#!/usr/bin/env python3
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
    ds = xr.open_dataset("METEOR_GNSS_IWV_20240813_000030_20240923_195300.nc", chunks={})
    ds.attrs["history"] += "; Converted to zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de"
    ds.to_zarr("METEOR_GNSS_IWV.zarr", mode="w", encoding=get_encoding(ds))


if __name__ == "__main__":
    main()
