import cdsapi

dataset = "reanalysis-carra-single-levels"
request = {
    "domain": "west_domain",
    "level_type": "surface_or_atmosphere",
    "variable": [
        "sea_ice_surface_temperature",
        "snow_density",
        "snow_on_ice_total_depth"
    ],
    "product_type": "analysis",
    "time": ["00:00"],
    "year": [
        "2003", "2004", "2005",
        "2006", "2007", "2008",
        "2009", "2010", "2011",
        "2012", "2013", "2014",
        "2015", "2016", "2017",
        "2018"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "data_format": "netcdf"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()