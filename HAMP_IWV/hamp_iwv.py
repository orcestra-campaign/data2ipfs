#!/usr/bin/env python3
import numcodecs
import xarray as xr


def get_compressor():
    numcodecs.blosc.set_nthreads(1)
    return numcodecs.Blosc("zstd", clevel=6)


def get_chunks(sizes):
    match tuple(sizes.keys()):
        case ("time", "frequency"):
            chunks = {
                "time": 4**8,
                "frequency": 5,
            }
        case ("time",):
            chunks = {
                "time": 4**9,
            }
        case (single_dim,):
            chunks = {single_dim: sizes[single_dim]}
        case _:
            chunks = {}

    return tuple((chunks[d] for d in sizes))


def get_encoding(dataset):
    return {
        var: {
            "compressor": get_compressor(),
            "chunks": get_chunks(dataset[var].sizes),
        }
        for var in dataset.variables
    }


def main():
    ds = xr.open_dataset(
        "/work/um0203/u301032/master_thesis/retrieved_data/PERCUSION_HAMP_IWV_IWP_LWP_TLWP.nc",
        chunks={},
    )
    ds = ds.assign_attrs(featureType="trajectory")
    ds.to_zarr(
        "PERCUSION_HAMP_IWV_IWP_LWP_TLWP.zarr",
        zarr_format=2,
        encoding=get_encoding(ds),
        mode="w",
    )


if __name__ == "__main__":
    main()
