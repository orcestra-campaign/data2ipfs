#!/usr/bin/env python3
#SBATCH --account=mh0010
#SBATCH --partition=compute
#SBATCH --time=01:00:00
import pathlib

import numcodecs
import xarray as xr


common_summary = (
    "To facilitate data analysis, Level 4 products are gridded on to 1, 2 or 3D Cartesian grids. Gridding was perfomed using the [Daisho](https://github.com/mmbell/Daisho.jl) Julia package developed by Michael Bell. Daisho uses a novel beam weighting instead of the traditional linear interpolation or distance weighting. The constant range and expanding azimuthal beam volume are considered to retain finer detail near the ship and more accurately represent the measured spatial resolution at longer range. Different weightings are used depending on the desired grid spacing and geometry.\n\n"
    'The data has two different missing data flags: `-32768` denotes "missing" data where the radar did not scan within the grid space, and `-9999` denotes "empty" data where the radar did scan but no data was recorded or was removed by quality control. The "missing" data includes regions such as the blanking sector or areas out of range of the radar. The "empty" data is determined by the MASK variable, which is either SQI for single polarization or PID for dual polarization. The "empty" `-9999` data can be considered as non-precipitating clear air, although the user is cautioned that this determination is range-dependent due to the sensitivity of the radar.\n\n'
    "Data are indexed at a time resolution of 5 minutes based on the start time of the volume. The exact start and end time of each volume are also recorded for more fine-scale time analysis. Volumes without a 0.5 elevation angle were not included in the rain rate product, but were included in the other 2D and 3D products.\n\n"
)

GLOBAL_ATTRS = {
    "PICCOLO_level4_rainrate_2D.nc": {
        "title": "Level 4 Gridded SEA-POL Radar Data (rainrate 2D)",
        "summary": common_summary
        + 'The "rainrate_2D" grids each PPI elevation angle independently with 1 km grid spacing in 2D out to 240 km. The 0.5 degree elevation angle is then used to construct a rain map as the lowest available rain rate estimate. If data is missing from the 0.5 angle, then successively higher angles are considered until an altitude of 1 km. At longer ranges, the 0.5 angle exceeds 1 km altitude but is still the lowest available rain rate and is retained. This product only contains the best available rain rate (Z-R or blended), the reflectivity, the hydrometeor identification, and the elevation angle used for the estimate. This product is recommended for the best available, lowest altitude rain rate.',
    },
    "PICCOLO_level4_volume_3D.nc": {
        "title": "Level 4 Gridded SEA-POL Radar Data (volume 3D)",
        "summary": common_summary
        + 'The "volume_3D" product grids all available elevation angles with a 1 km grid in 3D. This grid represents the best estimate of the 3D volumetric structure of the radar variables. The 1 km grid and weighting parameters were selected as an optimal combination for the volume out to 120 km range. The beam weighting retains some finer detail near the ship with coarser detail at longer range. The Cartesian gridding uses a transverse Mercator map projection. This product is recommended for general volumetric calculations, including constant altitude calculations.',
    },
    "PICCOLO_level4_rhi_2D.nc": {
        "title": "Level 4 Gridded SEA-POL Radar Data (rhi 2D)",
        "summary": common_summary,
    },
    "PICCOLO_level4_composite_2D.nc": {
        "title": "Level 4 Gridded SEA-POL Radar Data (composite 2D)",
        "summary": common_summary
        + 'The "composite_2D" product grids all available elevation angles with a 1 km grid in 2D using the maximum reflectivity in the vertical column. The other variables are associated with that maximum reflectivity. This product is recommended for comparison with model output composite reflectivity.',
    },
    "PICCOLO_level4_latlon_3D.nc": {
        "title": "Level 4 Gridded SEA-POL Radar Data (latlon 3D)",
        "summary": common_summary
        + 'The "latlon_3D" product grids all available elevation angles with a 0.02 degree grid spacing in the horizontal and 1 km grid spacing in the vertical. This resolution corresponds to an approximately 2 km horizontal grid, but uses degrees as the grid spacing to more directly compare with satellite or model products. Since the gridding is coarser than the volume product, it is recommended for comparison against satellite or model grids with similar resolution of 2 km.',
    },
    "PICCOLO_level4_qvp_1D.nc": {
        "title": "Level 4 Gridded SEA-POL Radar Data (qvp 1D)",
        "summary": common_summary
        + 'The "qvp_1D" gridding uses the 45 degree elevation angle sweeps near the ship and grids using a vertical 1D column with 100 meter resolution. This gridding preserves the native range resolution of the radar and is representative of a vertical profile near the ship. Due to the 45 degree angle, the higher altitudes are representative of a larger area, approximately corresponding to a circle with radius equivalent to the height. Variables are similar to the Level 3 CfRadial data.',
    },
}


def get_chunks(sizes):
    match tuple(sizes.keys()):
        case ("time", "Z", "Y", "X"):
            chunks = {
                "time": 16,
                "Z": 8,
                "Y": 128,
                "X": 128,
            }

        case ("time", "Y", "X"):
            chunks = {
                "time": 64,
                "Y": 128,
                "X": 128,
            }
        case ("time", "Z", "R"):
            chunks = {
                "time": 256,
                "Z": 73,
                "R": 46,
            }

        case ("time", z_or_r):
            chunks = {
                "time": 256,
                z_or_r: sizes[z_or_r],
            }

        case (single_dim,):
            chunks = {single_dim: sizes[single_dim]}
        case _:
            chunks = {}

    return tuple((chunks[d] for d in sizes))


def get_compressor():
    return numcodecs.Blosc("zstd", clevel=6)


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs

    return {
        var: {
            "compressor": get_compressor(),
            "chunks": get_chunks(dataset[var].sizes),
        }
        for var in dataset.variables
    }


for ncfile in pathlib.Path("./data").glob("*.nc"):
    print(ncfile)
    ds = xr.open_dataset(ncfile, chunks={"time": 256})
    ds.attrs.update(creator_name=ds.creator_name.replace(" and", ""))

    for varname, da in ds.variables.items():
        # Rechunk one-dimensional time series along time dimension.
        if varname not in ds.dims and da.dims == ("time",):
            ds[varname] = da.chunk(time=-1)

    ds.to_zarr(
        ncfile.with_suffix(".zarr"),
        mode="w",
        encoding=get_encoding(ds),
        zarr_format=2,
        compute=True,
    )
