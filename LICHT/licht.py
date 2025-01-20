#!/usr/bin/env python3
import numcodecs
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("time",):
            chunks = {
                "time": 2**18,
            }
        case ("time", "lim"):
            chunks = {
                "time": 2**18,
                "lim": 2,
            }
        case ("time", "alt"):
            chunks = {
                "time": 2**10,
                "alt": 242,
            }
        case ("time", "alt", "lim"):
            chunks = {
                "time": 2**10,
                "alt": 242,
                "lim": 2,
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
    ds_b = xr.open_mfdataset("**/LICHT-LIDAR_b*.nc", chunks={"time": -1, "alt": -1}).load()
    ds_b.attrs["history"] += "; converted to zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    ds_b.to_zarr("LICHT-LIDAR_b.zarr", mode="w", encoding=get_encoding(ds_b))

    ds_t = xr.open_mfdataset("**/LICHT-LIDAR_t*.nc", chunks={"time": -1, "alt": -1}).load()
    ds_t.attrs["history"] += "; converted to zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    ds_t.to_zarr("LICHT-LIDAR_t.zarr", mode="w", encoding=get_encoding(ds_t))

if __name__ == "__main__":
    main()
