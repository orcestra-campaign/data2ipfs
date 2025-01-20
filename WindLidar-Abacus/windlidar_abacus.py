# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: main
#     language: python
#     name: main
# ---

# +
import numcodecs
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("time", "height"):
            chunks = {
                "time": 2**12,
                "height": 64,
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
    ds_v0 = xr.open_mfdataset("/work/mh0010/ORCESTRA/raw/METEOR/WindLidar-Abacus/v0.0/nc_v0.0/*.nc", chunks={"time": -1})
    ds_v0.attrs["history"] += "; Converted to zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    ds_v0.chunk(time=-1).to_zarr("v0.0.zarr", encoding=get_encoding(ds_v0), mode="w")
    
    ds_v1 = xr.open_mfdataset("/work/mh0010/ORCESTRA/raw/METEOR/WindLidar-Abacus/v1.0/nc_v1.0/*.nc", chunks={"time": -1})
    ds_v1.attrs["history"] += "; Converted to zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    ds_v1.chunk(time=-1).to_zarr("v1.0.zarr", encoding=get_encoding(ds_v1), mode="w")
    
    ds_v2 = xr.open_mfdataset("/work/mh0010/ORCESTRA/raw/METEOR/WindLidar-Abacus/v2.0/nc_v2.0/*.nc", chunks={"time": -1})
    ds_v2.attrs["history"] += "; Converted to zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    ds_v2.chunk(time=-1).to_zarr("v2.0.zarr", encoding=get_encoding(ds_v2), mode="w")


if __name__ == "__main__":
    main()
