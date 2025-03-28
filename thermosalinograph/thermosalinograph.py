import fsspec
import numcodecs
import numpy as np
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("TIME",):
            chunks = {
                "TIME": 2**16,
            }
        case ("Depth",):
            chunks = {
                "Depth": 1,
            }
        case ("Depth", "TIME"):
            chunks = {
                "TIME": 2**16,
                "Depth": 1,
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
    cid = "Qmcc91KSJ18iZGzGzS1vcDfS2XGxVmhvRQtUEoeo6MwRSw"
    ds = xr.open_dataset(
        fsspec.open_local(f"simplecache::ipfs://{cid}"),
        engine="netcdf4",
        chunks={"time": -1},
    ).load()

    # Round to original temporal resolution of 1-min (see ASCII data)
    ds = ds.assign_coords(
        TIME=(ds.TIME + np.timedelta64(500, "ms")).astype("datetime64[m]")
    )
    ds.TIME.encoding["units"] = "minutes since 1970-01-01"
    ds.TIME.encoding["dtype"] = "<i4"

    ds.attrs["references"] = ds.attrs["references"].replace("\n", "")
    ds.attrs["license"] = "CC-BY-4.0"

    ds.to_zarr(
        "met_203_1_tsal.zarr", mode="w", encoding=get_encoding(ds), zarr_format=2
    )


if __name__ == "__main__":
    main()
