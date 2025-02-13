import glob
import pathlib

import numcodecs
import xarray as xr


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd")

    return {
        var: {
            "chunks": (2**18,),
            "compressor": codec,
        }
        if var not in dataset.dims
        else {"chunks": -1}
        for var in dataset.variables
    }


def merge_datasets(datasets, outfile):
    ds = xr.open_mfdataset(
        datasets,
        concat_dim="time",
        combine="nested",
        combine_attrs="drop_conflicts",
        chunks={"time": -1},
    ).sortby("time")

    ds.attrs["featureType"] = "trajectory"
    ds.attrs["license"] = "CC-BY-4.0"
    ds.attrs["references"] = ds.attrs["doi"]
    ds.attrs["platform"] = "ATR-42"
    ds.attrs["project"] = "ORCESTRA, MAESTRO"

    ds = ds.drop_vars("trajectory").load().dropna("time").chunk(time=2**18)

    ds.to_zarr(outfile, encoding=get_encoding(ds), mode="w")


if __name__ == "__main__":
    for source_dir in pathlib.Path().glob("MAESTRO-*"):
        print(source_dir)
        merge_datasets(source_dir.glob("*.nc"), source_dir.with_suffix(".zarr").name)
