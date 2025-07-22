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
        "single": "QmTCX3C1tiMaQyUfLkiazk2pqEbjPyKb2cc7Egzev3VAEK",
        "multi": "QmfJxdi75KNWr6bk9uDdFYzbRGxDf68NYzQ5cpBZpjjjhw",
    }

    # Collect datasets for single- and multi-pointing HATPRO products.
    for ds_name, root_cid in hatpro_datasets.items():
        fs = fsspec.filesystem("ipfs")
        hatpro_files = sorted(
            ["ipfs://" + item["name"] for item in fs.listdir(root_cid)]
        )

        # The HATPRO datasets have different height coordinates from August 22nd on.
        campaign_parts = [
            hatpro_files[:5],
            hatpro_files[5:],
        ]

        # Loop over 1st and 2nd campaign part
        for part, part_files in enumerate(campaign_parts, start=1):
            hatpro = xr.open_mfdataset(part_files, engine="zarr")
            hatpro.attrs["title"] += f" (Part {part})"
            hatpro.attrs["license"] = hatpro.attrs["license"].replace(" ", "-")
            hatpro.attrs["project"] = "BOW-TIE"
            hatpro.attrs["keywords"] = "HATPRO, Radiometer, Microwave"
            hatpro.attrs["featureType"] = "trajectoryProfile"

            hatpro.load().to_zarr(
                f"hatpro_{ds_name}_part{part}.zarr",
                encoding=get_encoding(hatpro),
                zarr_format=2,
                mode="w",
            )


if __name__ == "__main__":
    _main()
