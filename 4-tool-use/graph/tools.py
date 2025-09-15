from urllib.parse import urlencode

import requests
from langchain_core.tools import tool


@tool
def get_horoscope(sign):
    """
    Given a sign, return today's horoscope.
    """
    return f"{sign}: Next Tuesday you will befriend a baby otter."


@tool
def get_weather(city: str):
    """
    Given a Malaysian city, return today's weather forecast.
    """
    print(f"Actual city is {city}")
    city_query = urlencode({"contains": city + "@location__location_name"})

    base_url = f"https://api.data.gov.my/weather/forecast?{city_query}"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None


tools = [get_horoscope, get_weather]
