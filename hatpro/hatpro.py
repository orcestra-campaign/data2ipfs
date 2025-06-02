import glob

import numcodecs
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("time",):
            chunks = {
                "time": 2**18,
            }
        case ("height",):
            chunks = {
                "height": -1,
            }
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
    }


def _main():
    ncfiles = sorted(glob.glob("*.nc"))[5:]
    ds = xr.open_mfdataset(ncfiles, combine_attrs="drop_conflicts")
    ds.attrs["featureType"] = "trajectory"

    ds.attrs["title"] = (
        "Microwave radiometer (HATPRO) single-pointing measurements during METEOR cruise M203"
    )
    ds.attrs["keywords"] = "microwave, radiometer, water vapor"

    ds.attrs["creator_name"] = "Heike Kalesse-Los, Anna Trosits"
    ds.attrs["creator_email"] = "heike.kalesse@uni-leipzig.de, at58voge@studserv.uni-leipzig.de"

    ds.attrs["project"] = "ORCESTRA, BOW-TIE"
    ds.attrs["platform"] = "RV METEOR"

    ds.attrs["history"] = "Converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    ds.attrs["references"] = (
        "https://cloudnet.fmi.fi/collection/accaa493-1cab-44d1-b4ee-daf727dc5e84, https://hdl.handle.net/21.12132/3.3e013ad163c5451d"
    )
    ds.attrs["license"] = "CC-BY-4.0"

    ds.chunk(time=-1).to_zarr(
        "RV-METEOR_HATPRO-single.zarr", encoding=get_encoding(ds), mode="w"
    )


if __name__ == "__main__":
    _main()
