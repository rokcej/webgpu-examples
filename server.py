# Google Chrome WebGPU Origin Trial token:
# AkXOW6i/Qk3p0TM1XAi0kXUdwLNY8tZyeduts+g92KjzFttZfxgrp6jxQ9h+sUEsJWZWEol/S2JzVAHWY8L7zwAAAABHeyJvcmlnaW4iOiJodHRwOi8vbG9jYWxob3N0OjgwIiwiZmVhdHVyZSI6IldlYkdQVSIsImV4cGlyeSI6MTY1MjgzMTk5OX0=

import http.server
import sys

DEFAULT_PORT = 80

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
	extensions_map = {
        "": "application/octet-stream", # Default
		".html": 'text/html',
		".css": 'text/css',
		".js": 'application/javascript'
	}

	def end_headers(self):
		self.send_header("Origin-Trial", "AkXOW6i/Qk3p0TM1XAi0kXUdwLNY8tZyeduts+g92KjzFttZfxgrp6jxQ9h+sUEsJWZWEol/S2JzVAHWY8L7zwAAAABHeyJvcmlnaW4iOiJodHRwOi8vbG9jYWxob3N0OjgwIiwiZmVhdHVyZSI6IldlYkdQVSIsImV4cGlyeSI6MTY1MjgzMTk5OX0=")
		super().end_headers()


if __name__ == '__main__':
	with http.server.ThreadingHTTPServer(("", DEFAULT_PORT), CustomHTTPRequestHandler) as httpd:
		try:
			ip, port = httpd.server_address
			print(f"Server running on {ip}:{port}")
			httpd.serve_forever()
		except KeyboardInterrupt:
			print("\n[KeyboardInterrupt] Shutting down...")
			sys.exit(0)
