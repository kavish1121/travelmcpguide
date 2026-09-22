import requests

# Paste your Aviationstack API access key here
API_KEY = "my_key"
BASE_URL = "https://api.aviationstack.com/v1"


def test_flights():
    print("--- Testing Flights Endpoint ---")

    url = f"{BASE_URL}/flights"
    params = {
        "access_key": API_KEY,
        "limit": 2,  # Limits the response to 2 flights for testing
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code == 200 and "data" in data:
            print(f"Successfully retrieved {len(data['data'])} flights.")

            for flight in data["data"]:
                flight_date = flight.get("flight_date")
                flight_num = flight.get("flight", {}).get("iata")
                dep_airport = flight.get("departure", {}).get("airport")
                arr_airport = flight.get("arrival", {}).get("airport")
                status = flight.get("flight_status")

                print(
                    f"- Flight {flight_num} on {flight_date}: "
                    f"{dep_airport} -> {arr_airport} ({status})"
                )
        else:
            print(
                f"Error: {data.get('error', {}).get('message', 'Unknown error')}"
            )

    except Exception as e:
        print(f"An error occurred: {e}")


def test_airplanes():
    print("\n--- Testing Airplanes Endpoint ---")

    url = f"{BASE_URL}/airplanes"
    params = {
        "access_key": API_KEY,
        "limit": 2,  # Limits the response to 2 airplanes for testing
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code == 200 and "data" in data:
            print(f"Successfully retrieved {len(data['data'])} airplanes.")

            for plane in data["data"]:
                iata_type = plane.get("iata_code_short")
                registration = plane.get("registration_number")
                production_line = plane.get("production_line")
                age = plane.get("plane_age")

                print(
                    f"- Reg: {registration} | "
                    f"Type: {iata_type} | "
                    f"Model: {production_line} | "
                    f"Age: {age} years"
                )
        else:
            print(
                f"Error: {data.get('error', {}).get('message', 'Unknown error')}"
            )

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    if API_KEY == "YOUR_ACCESS_KEY_HERE":
        print(
            "Please replace 'YOUR_ACCESS_KEY_HERE' "
            "with your actual Aviationstack API key."
        )
    else:
        test_flights()
        test_airplanes()
