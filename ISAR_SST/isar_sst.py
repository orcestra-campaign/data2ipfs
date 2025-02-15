import xarray as xr


def main():
    ds = xr.open_mfdataset("*.nc", combine_attrs="drop_conflicts", decode_cf=False)

    # WTF...
    ds = ds.drop_vars("julian_day")
    ds.time.attrs["calendar"] = "proleptic_gregorian"
    ds = xr.decode_cf(ds)

    # Fix global attributes
    ds.attrs["featureType"] = ds.attrs.pop("feature_type")
    ds.attrs["source"] = ds.attrs["comment"]
    ds.attrs.pop("doi")
    ds.attrs["keywords"] = ", ".join(k.strip() for k in ds.attrs["keywords"].split(">"))
    ds.attrs["license"] = "CC-BY-SA-4.0"
    ds.attrs["platform"] = "RV METEOR"
    ds.attrs["project"] = "ORCESTRA, BOW-TIE, " + ds.attrs["project"]
    ds.attrs["references"] = ds.attrs["publisher_url"]

    ds.chunk(time=-1).to_zarr("ISAR_SST.zarr", mode="w")


if __name__ == "__main__":
    main()
