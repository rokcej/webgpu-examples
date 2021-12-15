# AkXOW6i/Qk3p0TM1XAi0kXUdwLNY8tZyeduts+g92KjzFttZfxgrp6jxQ9h+sUEsJWZWEol/S2JzVAHWY8L7zwAAAABHeyJvcmlnaW4iOiJodHRwOi8vbG9jYWxob3N0OjgwIiwiZmVhdHVyZSI6IldlYkdQVSIsImV4cGlyeSI6MTY1MjgzMTk5OX0=

from http import server

class MyHTTPRequestHandler(server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Origin-Trial", "AkXOW6i/Qk3p0TM1XAi0kXUdwLNY8tZyeduts+g92KjzFttZfxgrp6jxQ9h+sUEsJWZWEol/S2JzVAHWY8L7zwAAAABHeyJvcmlnaW4iOiJodHRwOi8vbG9jYWxob3N0OjgwIiwiZmVhdHVyZSI6IldlYkdQVSIsImV4cGlyeSI6MTY1MjgzMTk5OX0=")
        server.SimpleHTTPRequestHandler.end_headers(self)

if __name__ == '__main__':
    server.test(HandlerClass=MyHTTPRequestHandler, port=80)
