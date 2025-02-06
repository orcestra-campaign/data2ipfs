import fsspec
import numcodecs
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("TIME",):
            chunks = {
                "TIME": 2**16,
            }
        case ("TIME", "NO_OF_TEMPERATURES"):
            chunks = {
                "TIME": 2**16,
                "NO_OF_TEMPERATURES": 2,
            }
        case ("TIME", "NO_OF_SALINITIES"):
            chunks = {
                "TIME": 2**16,
                "NO_OF_SALINITIES": 2,
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
    cid = "QmPzRf5CWUdUZGz81FH56MPGZ2HUVBm8PTSchs1zuZWwdB"
    ds = xr.open_dataset(
        fsspec.open_local(f"simplecache::ipfs://{cid}"),
        engine="netcdf4",
        chunks={"time": -1},
    ).load()

    ds.attrs["featureType"] = "trajectory"
    ds.LATITUDE.attrs["units"] = "degrees_north"
    ds.LONGITUDE.attrs["units"] = "degrees_east"
    ds.TIME.encoding["units"] = "minutes since 1970-01-01"

    ds.attrs["title"] = (
        "Continuous thermosalinograph oceanography along RV METEOR cruise track M203"
    )
    ds.attrs["creator_name"] = ", ".join(ds.attrs["contributor_name"])
    ds.attrs["creator_email"] = ", ".join(ds.attrs["contributor_email"])
    ds.attrs["project"] = "ORCESTRA, BOW-TIE"
    ds.attrs["platform"] = "RV METEOR"
    ds.attrs["source"] = "thermosalinograph (TSG) system"
    ds.attrs["history"] = "Converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    ds.attrs["license"] = "CC-BY-4.0"

    ds.to_zarr(
        "thermosalinograph.zarr", mode="w", encoding=get_encoding(ds), zarr_format=2
    )


if __name__ == "__main__":
    main()
