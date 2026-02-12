import fsspec
import numcodecs
import numpy as np
import xarray as xr


def get_chunks(sizes):
    match tuple(sizes.keys()):
        case ("model_time", "model_height"):
            chunks = {
                "model_time": sizes["model_time"],
                "model_height": sizes["model_height"],
            }
        case ("time", "height"):
            chunks = {
                "time": 2**12,
                "height": 2**6,
            }
        case ("time",):
            chunks = {
                "time": 2**18,
            }
        case (single_dim,):
            chunks = {
                single_dim: sizes[single_dim],
            }
        case _:
            chunks = {}

    return tuple((chunks[d] for d in sizes))


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    compressor = numcodecs.Blosc("lz4")

    return {
        var: {
            "compressor": compressor,
            "chunks": get_chunks(dataset[var].sizes),
        }
        for var in dataset.variables
    }


def open_mfdataset_ipfs(
    root_cid,
    chunks={},
    concat_dim="time",
    combine_attrs="drop_conflicts",
    preprocess=lambda a: a,
):
    fs = fsspec.filesystem("ipfs")
    return xr.concat(
        [
            xr.open_dataset(
                "ipfs://" + item["name"], engine="zarr", chunks=chunks
            ).pipe(preprocess)
            for item in fs.listdir(root_cid)
        ],
        dim=concat_dim,
        combine_attrs=combine_attrs,
        data_vars="all",
    )


def round_datetime(ds):
    return ds.assign_coords(
        time=(ds.time.values + np.timedelta64(500, "ms"))
        .astype("datetime64[s]")
        .astype(ds.time.dtype)
    )


def drop_model_vars(ds):
    model_vars = [
        var
        for var in ds.variables
        if any(dim.startswith("model_") for dim in ds[var].dims)
    ]

    return ds.drop_vars(model_vars)


def _main():
    cloudnet_class = open_mfdataset_ipfs(
        "QmTj1JARz8xF35YpL3kjxHF5BGBEcoN3aVhNeEarXDQZYF"
    ).pipe(round_datetime)
    cloudnet_drop_reff = open_mfdataset_ipfs(
        "QmaxVaUUSyx2ouPMq4v2K87jZE4fekeP6eMncv8pbcNNKQ"
    ).pipe(round_datetime)
    cloudnet_crys_reff = open_mfdataset_ipfs(
        "Qmf8U3SgJ62faETgeZ5uRduECPNNfzQKAgMLY9wAL3dPog"
    ).pipe(round_datetime)
    cloudnet_iwc = open_mfdataset_ipfs(
        "QmRu2PQp9AWF2cagF5UvJjNZn77p3cyg4kZcWAeBq2QETk"
    ).pipe(round_datetime)
    cloudnet_lwc = open_mfdataset_ipfs(
        "QmbJnFZgubQh3j1Xco1Ex2fWEE1prixfH5d5V8wqC1pEC4"
    ).pipe(round_datetime)

    # Drop ECMWF variables for now, because of conflicting `model_time` and `model_height` coordinates.
    cloudnet_class_ecmwf = open_mfdataset_ipfs(
        "QmZ3Vpi28t2QjKUWnXmPPL2y4odRwSJRU7sQHpbU5WMmTj", preprocess=drop_model_vars
    )

    cloudnet = xr.merge(
        [
            cloudnet_drop_reff,
            cloudnet_crys_reff,
            cloudnet_iwc,
            cloudnet_lwc,
            cloudnet_class,
            cloudnet_class_ecmwf,
        ],
        compat="override",
        join="outer",
    )
    cloudnet.attrs["title"] = "Cloud radar and Cloudnet on RV Meteor during BOWTIE"
    cloudnet.attrs["license"] = cloudnet.attrs["license"].replace(" ", "-")
    cloudnet.attrs["project"] = "BOW-TIE"
    cloudnet.attrs["keywords"] = "Cloudnet, effective radius, droplet"
    cloudnet.attrs["featureType"] = "trajectoryProfile"

    cloudnet.load().to_zarr(
        "cloudnet.zarr",
        encoding=get_encoding(cloudnet),
        zarr_format=2,
        mode="w",
    )


if __name__ == "__main__":
    _main()
