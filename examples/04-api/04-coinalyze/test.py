import json
import requests
from datetime import datetime
from fintorch.utils.timestamp import to_timestamp

def main():
    default_params = {"api_key": "95b6d160-831b-4155-befe-73f6cb9a3836"}
    
    url = "https://api.coinalyze.net/v1/exchanges"
    params = default_params.copy()
    response = requests.get(url=url, params=params)
    results = json.loads(response.text)
    # print()
    # print("=" * 32)
    # for result in results:
    #     if "binance" == result["name"].lower():
    #         print(result)
            
    url = "https://api.coinalyze.net/v1/future-markets"
    params = default_params.copy()
    response = requests.get(url=url, params=params)
    results = json.loads(response.text)
    # print()
    # print("=" * 32)
    # for result in results:
    #     if "A" == result["exchange"]:
    #         print(result)
    
    url = "https://api.coinalyze.net/v1/long-short-ratio-history"
    url = "https://api.coinalyze.net/v1/open-interest-history"
    params = default_params.copy()
    params["symbols"] = "BTCUSDT_PERP.A"
    params["interval"] = "daily"
    params["from"] = int(to_timestamp(date="2021-01-01"))
    params["to"] = int(to_timestamp(date="2021-01-30"))
    response = requests.get(url=url, params=params)
    results = json.loads(response.text)
    print(results)
    result = results[0]
    for long_short_ratio in result["history"]:
        print(datetime.fromtimestamp(long_short_ratio["t"]), long_short_ratio["r"], long_short_ratio["l"], long_short_ratio["s"])
    
    
if __name__ == '__main__':
    main()