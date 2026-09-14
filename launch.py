"""Open the workspace only after its HTTP endpoint is ready."""
import json
import os
import threading
import time
import urllib.request
import webbrowser


def running(url):
    try:
        with urllib.request.urlopen(url + '/api/health', timeout=1) as response:
            return json.load(response).get('app') == 'homework-studio'
    except (OSError, ValueError):
        return False


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    url = f'http://127.0.0.1:{port}'
    if running(url):
        webbrowser.open(url)
    else:
        from server import app
        from waitress import serve

        def open_when_ready():
            for _ in range(50):
                if running(url):
                    webbrowser.open(url)
                    return
                time.sleep(0.2)

        threading.Thread(target=open_when_ready, daemon=True).start()
        print(f'Homework Studio: {url}\nPress Ctrl+C to stop.', flush=True)
        serve(app, host='127.0.0.1', port=port, threads=4)
