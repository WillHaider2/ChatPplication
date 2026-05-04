import os
import pandas as pd
from langchain.tools import tool
from geopy.distance import geodesic
from geopy.geocoders import OpenCage


@tool
def get_college_stat(college: str):
    """
    Use this tool to retrieve statistics, tuition, admission rates, and
    location information for a specific college
    Only Pull from schools listed Below, Ignore any other university:
- Alabama = "University of Alabama"
- Arkansas = "University of Arkansas"
- Auburn = "Auburn University"
- Florida = "University of Florida"
- Georgia = "University of Georgia"
- Kentucky = "University of Kentucky"
- LSU = "Louisiana State University"
- Mississippi State = "Mississippi State University"
- Missouri = "University of Missouri"
- Oklahoma = "University of Oklahoma"
- Ole Miss = "University of Mississippi"
- South Carolina = "University of South Carolina"
- Tennessee = "University of Tennessee"
- Texas = "University of Texas at Austin"
- Texas A&M = "Texas A&M University"
- Vanderbilt = "Vanderbilt University"
    """
    print(f"The agent is searching for data on: {college}")

    # Load the dataset you saved from the API
    data = pd.read_csv("Tool_files/full_college_data.csv")

    # Filter for the specific college name (Case-insensitive)
    result = data[data['name'].str.lower() == college.lower()]

    if result.empty:
        return f"College '{college}' not found. Please check the spelling."

    # Convert the matching row to a dictionary so the agent can read all stats
    return result.iloc[0].to_dict()


@tool
def get_distance(home_address: str, college: str):
    """
    MANDATORY: Use this tool whenever a user mentions a location, asks about proximity,
    travel, or the distance to a specific college.

    DIRECTIONS:
    1. If the user's 'home_address' (City, Zip, or State) is unknown, you MUST ask
       the user for it before providing any distance estimates.
    2. DO NOT estimate or 'ballpark' distances using internal knowledge.
    3. Input 'home_address' can be a City, Zip Code, or Full Address.
    4. Input 'college' must be the full name of the institution.

    This tool provides the only authoritative source for mileage in this conversation.
    """
    # 1. Load your dataset (now with lat/lon columns)
    print(f"🚨 CHAD IS TRYING TO RUN GET_DISTANCE FOR: {college}")
    data = pd.read_csv("Tool_files/full_college_data.csv")

    # 2. Find the college in the CSV
    college_row = data[data['name'].str.lower() == college.lower()]

    if college_row.empty:
        return f"I couldn't find {college} in my database to calculate distance."

    # Pull coordinates from CSV
    college_lat = college_row.iloc[0]['location.lat']
    college_lon = college_row.iloc[0]['location.lon']
    college_coords = (college_lat, college_lon)

    # 3. Geocode the USER'S home address
    # (Using OpenCage to find where the student lives)
    try:
        geocoder = OpenCage(api_key=os.getenv("OPENCAGE_API_KEY"))
        user_location = geocoder.geocode(home_address)

        if not user_location:
            return "I couldn't pin down that home address. Could you be more specific?"

        user_coords = (user_location.latitude, user_location.longitude)
        # This takes the two sets of (Lat, Lon) and calculates the miles
        miles = geodesic(user_coords, college_coords).miles
        # This sends the final answer back to Chad
        return f"{college} is {round(miles, 1)} miles away from {home_address}."
    except Exception as e:
        return f"There was an error connecting to the location service: {e}"



