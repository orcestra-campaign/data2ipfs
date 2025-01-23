import fsspec
import numcodecs
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("time",):
            chunks = {
                "time": 2**16,
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
    root = "QmSvZMKWETVTymrLN32cQZqjpN1xsQkWHFj3g5jfdzzbyi"
    ds = xr.open_dataset(
        fsspec.open_local(f"simplecache::ipfs://{root}"),
        engine="netcdf4",
        chunks={"time": -1},
    ).load()
    ds.attrs["history"] = "Converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"

    ds.attrs["title"] = "Ship information database (DVS DShip) of METEOR cruise M203"
    ds.attrs["creator_name"] = "Hans Segura"
    ds.attrs["creator_email"] = "hans.segura@mpimet.mpg.de"
    ds.attrs["project"] = "ORCESTRA, BOW-TIE"
    ds.attrs["platform"] = "RV METEOR"
    ds.attrs["source"] = "DVS DShip"
    ds.attrs["history"] = (
        "Converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    )
    ds.attrs["license"] = "CC-BY-4.0"

    ds.to_zarr("DShip.zarr", mode="w", encoding=get_encoding(ds), zarr_format=2)


if __name__ == "__main__":
    main()
