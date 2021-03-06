import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

# environment variables
os.environ['SPOTIPY_CLIENT_ID'] = ""
os.environ['SPOTIPY_CLIENT_SECRET'] = ""
os.environ['SPOTIPY_REDIRECT_URI'] = "http://127.0.0.1:9090"


auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)

# Sample code for testing
playlists = sp.user_playlists('spotify')
while playlists:
    for i, playlist in enumerate(playlists['items']):
        print("%4d %s %s" % (i + 1 + playlists['offset'], playlist['uri'],  playlist['name']))
    if playlists['next']:
        playlists = sp.next(playlists)
    else:
        playlists = None
