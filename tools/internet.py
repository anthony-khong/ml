import requests

def is_on():
    try:
        requests.get('http://216.58.192.142', timeout=1)
        return True
    except requests.ConnectionError as err: 
        reason = err.args[0].reason
        print(reason)
        return False

def is_off():
    return not is_on()
