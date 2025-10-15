# Copyright (c) Sony AI Inc.
# All rights reserved.
"""Module containing utilities used in all FHIBE datasets.

This module contains a function to fix the location_country
attribute in the annotation dataframes used for both FHIBE and FHIBE-face. 
This attribute was written in by subjects and contains various misspellings.
"""

loc_country_name_mapping = {
    "Abgola": "Angola",
    "Abuja": "Nigeria",
    "Argentiina": "Argentina",
    "Australie": "Australia",
    "Autsralia": "Australia",
    "Auustralia": "Australia",
    "Bahamas, The": "Bahamas",
    "Caanada": "Canada",
    "Canadad": "Canada",
    "French": "France",
    "Hanoi Vietnam": "Viet Nam",
    "Ho Chi Min": "Viet Nam",
    "Hong Kong": "China, Hong Kong Special Administrative Region",
    "I Go": None,
    "Italiana": "Italy",
    "Keenya": "Kenya",
    "Kenyan": "Kenya",
    "Kiambu": "Kenya",
    "Lagos": "Nigeria",
    "Lceland": "Iceland",
    "Mexican": "Mexico",
    "Micronesia": "Micronesia (Federated States of)",
    "Mironesi": "Micronesia (Federated States of)",
    "Mironesia": "Micronesia (Federated States of)",
    "Morroco": "Morocco",
    "Muranga": "Kenya",
    "Nairobi Nairobi": "Kenya",
    "Netherlands": "Netherlands (Kingdom of the)",
    "Nigerian": "Nigeria",
    "Nigeriia": "Nigeria",
    "Niheria": "Nigeria",
    "Nugeria": "Nigeria",
    "Nyari": "Kenya",
    "Owow Disable Abilities Off Level Up": None,
    "Pakisan": "Pakistan",
    "Pakisatn": "Pakistan",
    "Pakistain": "Pakistan",
    "Paksitan": "Pakistan",
    "Phillipines": "Philippines",
    "Punjab": "Pakistan",
    "South Afica": "South Africa",
    "South Afria": "South Africa",
    "South African": "South Africa",
    "Southern Africa": "South Africa",
    "South Korea": "Republic of Korea",
    "Tanzania": "United Republic of Tanzania",
    "Trinidad And Tobago": "Trinidad and Tobago",
    "Turkey": "Türkiye",
    "Ua": "Ukraine",
    "Uae": "United Arab Emirates",
    "Ugnd": "Uganda",
    "Uk": "United Kingdom of Great Britain and Northern Ireland",
    "United Kingdom": "United Kingdom of Great Britain and Northern Ireland",
    "Ukaine": "Ukraine",
    "United States": "United States of America",
    "Usa": "United States of America",
    "Venezuela": "Venezuela (Bolivarian Republic of)",
    "Veitnam": "Viet Nam",
    "Vienam": "Viet Nam",
    "Vietam": "Viet Nam",
    "Vietnam": "Viet Nam",
    "Vietname": "Viet Nam",
    "Viietnam": "Viet Nam",
    "Vitenam": "Viet Nam",
    "Vitnam": "Viet Nam",
    "Viwtnam": "Viet Nam",
}


def fix_location_country(country: str) -> str | None:
    """Format the location_country attribute string.

    Some countries are misspelled or inconsistently formatted.

    Args:
        country: The original string annotation

    Return:
        The re-formatted string
    """
    if country in loc_country_name_mapping:
        return loc_country_name_mapping[country]
    country_fmt = country.strip().title()
    if country_fmt in loc_country_name_mapping:
        return loc_country_name_mapping[country_fmt]
    else:
        return country_fmt
