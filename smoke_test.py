import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

def run_smoke_tests():
    print("Executing post-deployment smoke tests...")
    try:
        res = requests.get(f"{BASE_URL}/health")
        if res.status_code == 200 and res.json().get("status") == "healthy":
            print("Health check endpoint PASSED")
        else:
            print("Health check FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"Could not connect to service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_tests()