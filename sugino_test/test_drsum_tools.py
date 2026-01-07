import drsum_tools
import json

def test_get_database_list():
    print("Testing get_database_list...")
    try:
        result = drsum_tools.get_database_list(limit=5)
        print("Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_get_database_list()
