#!/usr/bin/env python3
import numcodecs
import xarray as xr


def get_chunks(dimensions):
    if "DEPTH" in dimensions:
        chunks = {
            "DEPTH": 4,
            "TIME": 2**16,
        }
    else:
        chunks = {
            "TIME": 2**16,
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
    ncfiles = (
        "met_203_vmadcp_38khz.nc",
        "met_203_vmadcp_75khz.nc",
    )

    for ncfile in ncfiles:
        ds = xr.open_dataset(ncfile, chunks={})
        ds.to_zarr(ncfile.replace(".nc", ".zarr"), mode="w", encoding=get_encoding(ds))


if __name__ == "__main__":
    main()
