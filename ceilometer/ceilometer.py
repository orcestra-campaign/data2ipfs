#!/usr/bin/env python3
import numcodecs
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("time",):
            chunks = {
                "time": 2**18,
            }
        case ("time", "range"):
            chunks = {
                "time": 2**10,
                "range": 2**8,
            }
        case ("time", "range_hr"):
            chunks = {
                "time": 2**15,
                "range_hr": 8,
            }
        case ("time", "layer"):
            chunks = {
                "time": 2**17,
                "layer": 3,
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


def main():
    ds = xr.open_mfdataset("/work/mh0010/ORCESTRA/raw/METEOR/ceilometer/*/*.nc")
    ds.chunk(time=2**18).to_zarr("CHM170158.zarr", encoding=get_encoding(ds), mode="w")


if __name__ == "__main__":
    main()
