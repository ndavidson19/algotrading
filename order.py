import requests

# Replace the placeholder values with your TD Ameritrade API key and access token
api_key = "your_api_key"
access_token = "your_access_token"

# Set the base URL for the API
base_url = "https://api.tdameritrade.com"

# Set the parameters for the order
ticker = "TICKER"
order_type = "buy" # or "sell"
quantity = 100
price = y_pred # Use the predicted stock price as the order price

# Create the endpoint URL
endpoint_url = f"{base_url}/v1/accounts/{account_id}/orders"

# Set the request headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {access_token}"
}

# Set the request body
body = {
    "orderType": order_type,
    "session": "NORMAL",
    "duration": "DAY",
    "orderStrategyType": "SINGLE",
    "orderLegCollection": [
        {
            "instruction": order_type,
            "quantity": quantity,
            "instrument": {
                "symbol": ticker,
                "assetType": "EQUITY"
            }
        }
    ],
    "price": price
}

# Make the request to place the order
response = requests.post(endpoint_url, headers=headers, json=body)
