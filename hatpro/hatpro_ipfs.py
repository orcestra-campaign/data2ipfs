import fsspec
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
    hatpro_datasets = {
        "single": "QmZHPTnWvBnixBrKg1617TurnCgmwiDd1pCTcBCg5ZR4fV",
        "multi": "QmaUkbMEDyVvEXTKVXrawHX3UMZeyCjwY5zqmuyJqiuUaP",
    }

    # Collect datasets for single- and multi-pointing HATPRO products.
    for ds_name, root_cid in hatpro_datasets.items():
        fs = fsspec.filesystem("ipfs")
        hatpro_files = sorted(
            ["ipfs://" + item["name"] for item in fs.listdir(root_cid)]
        )

        hatpro = xr.open_mfdataset(hatpro_files, engine="zarr")
        hatpro.attrs["license"] = hatpro.attrs["license"].replace(" ", "-")
        hatpro.attrs["project"] = "BOW-TIE"
        hatpro.attrs["keywords"] = "HATPRO, Radiometer, Microwave"
        hatpro.attrs["featureType"] = "trajectoryProfile"

        hatpro.load().to_zarr(
            f"hatpro_{ds_name}.zarr",
            encoding=get_encoding(hatpro),
            zarr_format=2,
            mode="w",
        )


if __name__ == "__main__":
    _main()
