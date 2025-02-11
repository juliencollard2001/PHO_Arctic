import cdsapi

dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
        "sea_surface_temperature",
        "surface_latent_heat_flux",
        "surface_net_solar_radiation",
        "surface_sensible_heat_flux",
        "snow_albedo",
        "snow_density",
        "snow_depth",
        "snow_evaporation",
        "snowfall",
        "snowmelt",
        "temperature_of_snow_layer",
        "total_column_snow_water"
    ],
    "year": [
        "2003", "2004", "2005",
        "2006", "2007", "2008",
        "2009", "2010", "2011",
        "2012", "2013", "2014",
        "2015", "2016", "2017",
        "2018", "2019", "2020",
        "2021", "2022", "2023"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [90, -180, 60, 180]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
