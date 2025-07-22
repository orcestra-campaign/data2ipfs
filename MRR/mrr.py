import numcodecs
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


def _main():
    mrr_rainflag = xr.open_dataset(
        "ipfs://QmaBppGnHXuDtwkv1RdQwdtAqGhJ1xpTCTrz8bgo9bJwfR/mrr_rainflag.zarr",
        engine="zarr",
        chunks={},
    )
    mrr_rainflag.attrs["license"] = mrr_rainflag.attrs["license"].replace(" ", "-")
    mrr_rainflag.attrs["project"] = "BOW-TIE"
    mrr_rainflag.attrs["featureType"] = "trajectory"

    mrr_rainflag.chunk(time=-1).to_zarr(
        "mrr_rainflag.zarr",
        encoding=get_encoding(mrr_rainflag),
        zarr_format=2,
        mode="w",
    )


if __name__ == "__main__":
    _main()
