# twottle
A web gui for browsing twitch.tv streams and videos on demand. Search for categories or users, and view followed streamers' activity. Requires user login.

## Installation
Works on all platforms (Win/Mac/Linux/BSD)

### Dependencies
- `python >=3.10` (Can relax requirement upon request)
- `streamlink` (via pip/binaries/brew/source explained [here](https://streamlink.github.io/install.html#))
- multimedia player listed [here](https://streamlink.github.io/players.html#player-compatibility)
  - Windows: player must be in system PATH

### PyPi:

`pip install twottle`

### From source:
- Follow dependencies listed below
- Build requires [`flit`](https://flit.readthedocs.io/en/latest/)
- Clone repository and run `flit install` in root directory


## Usage
```
twottle [-h] [--reset | --logout | -c | -d | -v]

options:
  -h, --help        show this help message and exit
  --reset           reset config file
  --logout          remove user from app, prompt login again
  -c, --clear-data  remove all user data and cache
  -d, --dump-cache  clear cache, stay logged in.
  -v, --version     show program's version number and exit
```
Run `twottle` to begin. On first startup you will be prompted to login and will be redirected to Twitch oauth.
