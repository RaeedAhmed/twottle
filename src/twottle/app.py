import argparse
import asyncio
import logging
import platform
import shlex
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from subprocess import DEVNULL, Popen
from typing import NamedTuple

import bottle as bt
import httpx
import peewee as pw
import waitress

import twottle

SYSTEM = platform.system().lower()  # Current OS


# Configuring file paths for referencing
module_path = Path(__file__).absolute().parent  # src/twottle
# tells bottle where to find html files
bt.TEMPLATE_PATH.insert(0, str(module_path / "views"))
static_path = module_path / "static"  # server static files e.g styles.css
# User/Category image cache
cache_path = module_path / "cache"
user_cache = cache_path / "users"
games_cache = cache_path / "games"


def mk_cache_dir() -> None:
    """
    Ensure image paths are valid when downloading/accessing
    """
    user_cache.mkdir(exist_ok=True, parents=True)
    games_cache.mkdir(exist_ok=True, parents=True)

# +--------------------------------╔═════════╗-------------------------------+ #
# |::::::::::::::::::::::::::::::::║ Logging ║:::::::::::::::::::::::::::::::| #
# +--------------------------------╚═════════╝-------------------------------+ #


logger = logging.getLogger(__name__)


def set_logger(level: int) -> None:
    """
    Create logger with stdout and file streams
    level of stdout is set by argparse (default level: logging.INFO)
    """
    # Logger object
    logger.setLevel(logging.DEBUG)
    # Create handlers for console and file
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(module_path / "debug.log", mode="w")
    c_handler.setLevel(level)
    f_handler.setLevel(logging.DEBUG)
    # Format for console and file
    c_format = logging.Formatter("\n%(message)s")
    f_format = logging.Formatter(
        fmt="\n%(asctime)s\nLine %(lineno)d: %(funcName)s()\n%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


# +--------------------------------╔════════╗--------------------------------+ #
# |::::::::::::::::::::::::::::::::║ Config ║::::::::::::::::::::::::::::::::| #
# +--------------------------------╚════════╝--------------------------------+ #


class Config(ConfigParser):
    path = module_path / "config.ini"

    def update(self):
        with open(Config.path, "w") as file:
            self.write(file)

    def reset(self):
        self["USER"].clear()
        self.update()
        logger.info("Reset settings")

    def apply(self, formdata: bt.FormsDict) -> list[str]:
        """
        Parse form data from /settings and check for changed or default values
        """
        settings = {k: v for k, v in dict(formdata).items() if v}
        changes = []
        for k, v in settings.items():
            old = self["USER"][k]
            if self.defaults()[k] == v:
                self.remove_option("USER", k)
            else:
                self["USER"][k] = v
            if old != v:
                changes.append(change := f"{k}:\n{old} -> {v}")
                logger.info(change)
        self.update()
        return changes


# init config
config = Config()
config.read(Config.path)

# routes that don't need to check for user/cache
static_pages = [
    "authenticate",
    "config",
    "settings",
    "error",
    "static/",
    "login",
    "cache/",
    "watch",
    "favicon",
]

# abort page
back_script = """<script>setTimeout(function () { window.history.back() });</script>"""


# +-------------------------------╔══════════╗-------------------------------+ #
# |:::::::::::::::::::::::::::::::║ Database ║:::::::::::::::::::::::::::::::| #
# +-------------------------------╚══════════╝-------------------------------+ #

db = pw.SqliteDatabase(module_path / "data.db")


class BaseModel(pw.Model):
    class Meta:
        database = db


class User(BaseModel):
    id = pw.IntegerField()
    login = pw.TextField()
    display_name = pw.TextField()
    profile_image_url = pw.TextField()
    access_token = pw.TextField()


class Streamer(BaseModel):
    id = pw.IntegerField(primary_key=True)
    login = pw.TextField()
    display_name = pw.TextField()
    broadcaster_type = pw.TextField(default="user")  # If not partner/affiliate
    description = pw.TextField(default="Twitch streamer")  # Empty description
    profile_image_url = pw.TextField()
    followed = pw.BooleanField(default=False)


class Game(BaseModel):
    id = pw.IntegerField(primary_key=True)
    name = pw.TextField()
    box_art_url = pw.TextField()


# +------------------------------╔═══════════╗-------------------------------+ #
# |::::::::::::::::::::::::::::::║ Processes ║:::::::::::::::::::::::::::::::| #
# +------------------------------╚═══════════╝-------------------------------+ #


class Media:
    procs: list = []

    def __init__(self, command: list[str], info: bt.FormsDict):
        self.runtime = Popen(command, stdout=DEVNULL)
        info.append("pid", self.runtime.pid)
        self.info = info
        logger.debug(f"proc command: {command}\nproc info: {dict(info)}")
        Media.procs.append(self)

    @classmethod
    def wipe(cls):
        procs = list(cls.procs)  # Temp list of procs as Media.procs is updated
        for proc in procs:
            end_proc(proc)

    @classmethod
    def update(cls):
        to_dump = []
        for proc in cls.procs:
            if proc.runtime.poll() is not None:  # None indicates running
                to_dump.append(proc)
        for proc in to_dump:
            end_proc(proc)

    @classmethod
    def kill(cls, pid: int):
        for proc in cls.procs:
            if proc.info.pid == pid:
                end_proc(proc)


def end_proc(proc: Media):
    if SYSTEM == "windows" and proc.runtime.poll() is None:
        Popen(f"TASKKILL /F /PID {proc.runtime.pid} /T", stdout=DEVNULL)
    else:
        if proc.runtime.poll() is None:
            proc.runtime.terminate()
    Media.procs.remove(proc)
    logger.debug(f"Terminating process {dict(proc.info)}")


# +----------------------------╔═════════════════╗---------------------------+ #
# |::::::::::::::::::::::::::::║ Data Containers ║:::::::::::::::::::::::::::| #
# +----------------------------╚═════════════════╝---------------------------+ #


class Image(NamedTuple):
    id: str
    url: str


class Stream(NamedTuple):
    id: str  # Stream ID
    user_id: str
    user_login: str
    user_name: str  # Display Name
    game_id: str
    type: str  # 'live' or ''
    title: str
    viewer_count: str
    thumbnail_url: str  # -480x270
    profile_image_url: str  # from user_id
    uptime: str  # time_elapsed(started_at)
    game_name: str = "Streaming"
    box_art_url: str = "https://static-cdn.jtvnw.net/ttv-static/404_boxart.jpg"


class Vod(NamedTuple):
    id: str
    user_id: str
    user_login: str
    user_name: str
    title: str
    description: str
    created_at: str
    url: str
    thumbnail_url: str
    view_count: str
    duration: str


class Clip(NamedTuple):
    id: str
    url: str
    title: str
    thumbnail_url: str
    game_id: str
    vod_link: str
    time_since: str
    view_count: str
    duration: str
    game_name: str = "Streaming"  # if no game id
    box_art_url: str = "https://static-cdn.jtvnw.net/ttv-static/404_boxart.jpg"


class SearchResult(NamedTuple):
    model: str
    results: list[pw.ModelSelect]


# +-----------------------------╔══════════════╗-----------------------------+ #
# |:::::::::::::::::::::::::::::║ API Requests ║:::::::::::::::::::::::::::::| #
# +-----------------------------╚══════════════╝-----------------------------+ #


class Helix:
    client_id = "o232r2a1vuu2yfki7j3208tvnx8uzq"
    redirect_uri = "http://localhost:8080/authenticate"
    app_scopes = "user:edit+user:read:follows+user:read:subscriptions"
    base = "https://api.twitch.tv/helix"
    oauth = (
        "https://id.twitch.tv/oauth2/authorize?client_id="
        f"{client_id}&redirect_uri={redirect_uri}"
        f"&response_type=token&scope={app_scopes}"
        "&force_verify=true"
    )

    @staticmethod
    def headers():
        return {
            "Client-ID": Helix.client_id,
            "Authorization": f"Bearer {User.get().access_token}",
        }


class Request:
    def __init__(self, endpoint: str, params: dict):
        transport = httpx.HTTPTransport(retries=3)
        self.params = params
        self.session = httpx.Client(
            base_url=Helix.base + endpoint,
            headers=Helix.headers(),
            params=params,
            transport=transport,
        )

    def json(self) -> dict:
        try:
            with self.session as session:
                data: dict = session.get("").json()
        except httpx.RequestError:
            return {}
        return data

    def get_iter(self) -> list[dict]:
        results = []
        with self.session as session:
            while True:
                resp = session.get("", params=self.params).json()
                data = resp["data"]
                if not data:
                    break
                results += data
                if not resp["pagination"]:
                    break
                pagination = resp["pagination"]["cursor"]
                self.params["after"] = pagination
        return results


class AsyncRequest:
    def __init__(self, endpoint: str):
        transport = httpx.AsyncHTTPTransport(retries=3)
        self.session = httpx.AsyncClient(
            base_url=Helix.base + endpoint, headers=Helix.headers(), transport=transport
        )

    async def get(self, ids: list) -> list[dict]:
        async with self.session:
            resps = await asyncio.gather(
                *(self.session.get(f"?id={i}") for i in ids)
            )
        return [item["data"][0] for resp in resps if (item := resp.json())]

    async def get_batch(self, ids: list, id_key="id") -> list[dict]:
        id_lists = [ids[x: x + 100] for x in range(0, len(ids), 100)]
        async with self.session:
            resps = await asyncio.gather(
                *(
                    self.session.get(
                        "?" + "&".join([f"{id_key}={i}" for i in idlist]))
                    for idlist in id_lists
                )
            )
        data = []
        [data.extend(datum)
         for resp in resps if (datum := resp.json()["data"])]
        return data


# +--------------------------------╔═══════╗---------------------------------+ #
# |::::::::::::::::::::::::::::::::║ Tests ║:::::::::::::::::::::::::::::::::| #
# +--------------------------------╚═══════╝---------------------------------+ #


def check_user() -> None:
    logger.debug("Checking user")
    if db.table_exists("user") is False or User.get_or_none() is None:
        logger.debug("No user table")
        bt.redirect("/login")


def check_cache() -> None:
    logger.debug("Checking cache")
    if (Streamer.table_exists() or Game.table_exists()) is False:
        logger.debug("No cache found")
        db.create_tables([Streamer, Game], safe=True)
        follows = get_follows()
        logger.debug(f"Following {len(follows)} channels")
        cache(follows, Streamer)
        db.execute(Streamer.update(followed=True))


# +------------------------------╔════════════╗------------------------------+ #
# |::::::::::::::::::::::::::::::║ Fetch Data ║::::::::::::::::::::::::::::::| #
# +------------------------------╚════════════╝------------------------------+ #


def get_user(access_token: str) -> None:
    headers = {"Client-ID": Helix.client_id,
               "Authorization": f"Bearer {access_token}"}
    endpoint = f"{Helix.base}/users"
    user: dict = httpx.get(endpoint, headers=headers,
                           timeout=None).json()["data"][0]
    logger.info(f"User {user['display_name']}")
    user["access_token"] = access_token
    User.create_table()
    User.create(**user)


def get_follows() -> set[int]:
    endpoint = "/users/follows"
    params = {"from_id": User.get().id, "first": "100"}
    resp = Request(endpoint, params).get_iter()
    return {int(follow["to_id"]) for follow in resp}


def update_follows() -> None:
    follows = get_follows()
    cache(follows, Streamer)
    streamers: list[Streamer] = [streamer for streamer in Streamer.select()]
    for streamer in streamers:
        sid = streamer.id
        if (sid in follows and streamer.followed is not True) or (
            sid not in follows and streamer.followed is True
        ):
            logger.info(
                f"{'un'*streamer.followed}following {streamer.display_name}")
            db.execute(
                Streamer.update(followed=not streamer.followed).where(
                    Streamer.id == sid
                )
            )


def get_followed_streams() -> list[dict]:
    follows = {
        streamer.id
        for streamer in Streamer.select().where(Streamer.followed == True).execute()
    }
    streams = asyncio.run(AsyncRequest(
        "/streams").get_batch(list(follows), id_key="user_id"))
    return streams


def format_streams(streams: list[dict]) -> list[Stream]:
    with ThreadPoolExecutor() as tp:
        futures = []
        for type_id, model in {"game_id": Game, "user_id": Streamer}.items():
            ids = {int(i) for stream in streams if (i := stream[type_id])}
            futures.append(tp.submit(cache(ids, model)))
        for future in as_completed(futures):
            future.done()
    for stream in streams:
        channel = Streamer.get_by_id(stream["user_id"])
        stream["profile_image_url"] = channel.profile_image_url
        stream["uptime"] = time_elapsed(stream["started_at"])
        stream["thumbnail_url"] = stream["thumbnail_url"].replace(
            "-{width}x{height}", "-480x270"
        )
        if gid := stream["game_id"]:
            game = Game.get_or_none(Game.id == gid)
            if game:
                stream["box_art_url"] = game.box_art_url
        if not gid or not game:
            stream["game_name"] = "Streaming"
    streams_fmt: list[Stream] = [
        Stream(**{k: v for k, v in stream.items() if k in Stream._fields})
        for stream in streams
    ]
    streams_fmt.sort(key=lambda stream: stream.viewer_count, reverse=True)
    return streams_fmt


# +------------------------------╔═════════════╗-----------------------------+ #
# |::::::::::::::::::::::::::::::║ Route Hooks ║:::::::::::::::::::::::::::::| #
# +------------------------------╚═════════════╝-----------------------------+ #


@bt.hook("before_request")
def _connect():
    db.connect()
    if not any(part in bt.request.path for part in static_pages):
        logger.debug(f"Start {bt.request.fullpath}")
        check_user()
        check_cache()


@bt.hook("after_request")
def _close():
    if not any(part in bt.request.path for part in static_pages):
        logger.debug(f"Close {bt.request.fullpath}")
    if not db.is_closed():
        db.close()


# +--------------------------╔════════════════════╗--------------------------+ #
# |::::::::::::::::::::::::::║ Application Routes ║::::::::::::::::::::::::::| #
# +--------------------------╚════════════════════╝--------------------------+ #


@bt.route("/")
def index():
    update_follows()
    streams = format_streams(get_followed_streams())
    return bt.template("index.html",
                       user=User.get_or_none(),
                       streams=streams,
                       config=config["USER"])


@bt.route("/login")
def login():
    return bt.template("login.html", oauth=Helix.oauth)


@bt.route("/authenticate")
def authenticate():
    access_token: str | None
    if access_token := bt.request.query.get("access_token"):
        get_user(access_token)
        bt.redirect("/")
    return bt.template("base.html", verification=True)


@bt.route("/settings", method=["GET", "POST"])
def settings():
    changes = []
    if bt.request.method == "POST":
        if bt.request.forms.get("reset"):
            config.reset()
            changes = ["settings reset"]
        else:
            changes = config.apply(bt.request.forms)
    return bt.template("settings.html", user=config["USER"], changes=changes)


@bt.route("/watch")
def watch():
    watch_video(bt.request.query)
    return back_script


@bt.route("/watching", method=["GET", "POST"])
def watching():
    if bt.request.method == "POST":
        if pid := bt.request.forms.get("pid"):
            Media.kill(int(pid))
        if bt.request.forms.get("wipe"):
            Media.wipe()
        return bt.redirect("/watching")
    Media.update()
    return bt.template("watching.html", procs=Media.procs)


@bt.route("/search")
def search():
    def search_results(query: str, endpoint: str):
        model = Game if "categories" in endpoint else Streamer
        resp = Request(endpoint, {"query": query, "first": 5}).json()
        if results := resp.get("data"):
            ids = {int(result["id"]) for result in results}
            cache(ids, model)
            return SearchResult(endpoint, model.select().where(model.id.in_(ids)))
        else:
            return None

    query = bt.request.query.get("query")
    if query:
        req = partial(search_results, query)

        with ThreadPoolExecutor() as tp:
            results = list(
                tp.map(req, ["/search/channels", "/search/categories"]))
    else:
        results = []
    return bt.template("search.html", query=query, results=results)


@bt.route("/static/<filename:path>")
def send_static(filename):
    return bt.static_file(filename, root=str(static_path))


@bt.route("/cache/<filename:path>")
def send_image(filename):
    return bt.static_file(filename, root=str(cache_path), mimetype="image/jpeg")


@bt.route("/u/<channel>")
@bt.route("/u/<channel>/<mode>")
def channel(channel, mode="default", data=None, stream=None):
    streamer = Streamer.get_or_none(Streamer.login == channel)
    if not streamer:
        data = Request("/users", {"login": f"{channel}"}).json().get("data")
        if data:
            uid = data[0]["id"]
            cache({uid}, Streamer)
            streamer = Streamer.get_by_id(uid)
        else:
            msg = f"User '{channel}' not found"
            return bt.template("error.html", message=msg)

    if mode == "vods":
        vod_type = bt.request.query.get("vod_type") or "archive"
        params = add_page({"user_id": streamer.id, "type": vod_type})
        resp = Request("/videos", params).json()
        if not resp.get("data"):
            return back_script
        vods = process_vods(resp["data"])
        pagination = resp["pagination"]["cursor"]
        return bt.template(
            "channel.html",
            channel=streamer,
            vods=vods,
            vod_type=vod_type,
            pagination=pagination,
            config=config["USER"],
        )

    elif mode == "clips":
        started_at = bt.request.query.get("started_at")
        ended_at = bt.request.query.get("ended_at")
        offset = "T00:00:00Z"
        params = add_page({"broadcaster_id": streamer.id,
                          "first": 30, "started_at": started_at+offset, "ended_at": ended_at+offset})
        resp = Request("/clips", params).json()
        if not (clips := resp["data"]):
            return back_script
        clips = process_clips(clips)
        clips.sort(key=lambda clip: clip.view_count, reverse=True)
        pagination = resp["pagination"]["cursor"]
        return bt.template(
            "channel.html",
            channel=streamer,
            clips=clips,
            started_at=started_at,
            ended_at=ended_at,
            pagination=pagination,
            config=config["USER"],
        )

    elif mode == "default":
        if stream_data := Request("/streams", {"user_id": streamer.id}).json()["data"]:
            stream = format_streams([stream_data[0]])[0]
        return bt.template("channel.html",
                           channel=streamer,
                           stream=stream,
                           config=config["USER"])


@bt.route("/c")
@bt.route("/c/<category_id>")
def category(category_id=None):
    if not category_id:
        resp = Request("/games/top", {"first": 100}).json().get("data")
        ids = {int(game["id"]) for game in resp}
        cache(ids, Game)
        games = [Game.get_by_id(int(game["id"])) for game in resp]
        return bt.template("games.html", games=games)
    if category_id == "all":
        params = {"first": 30}
        game = None
    else:
        game = Game.get_or_none(Game.id == category_id)
        if not game:
            data = Request("/games", {"id": category_id}).json().get("data")
            if data:
                gid = data[0]["id"]
                cache({gid}, Game)
                game = Game.get_by_id(gid)
            else:
                msg = "Game not found"
                return bt.template("error.html", message=msg)
        params = {"game_id": game.id, "first": 30}
    params = add_page(params)
    resp = Request("/streams", params).json()
    if not resp.get("data"):
        return back_script
    streams = format_streams(resp["data"])
    pagination = resp["pagination"]["cursor"]
    return bt.template(
        "streams.html",
        game=game,
        streams=streams,
        pagination=pagination,
        config=config["USER"],
    )


@bt.route("/following")
def following():
    follows = (
        Streamer.select()
        .where(Streamer.followed == True)
        .order_by(pw.fn.LOWER(Streamer.display_name))
    )
    return bt.template("following.html", follows=follows)


@bt.error(400)
@bt.error(403)
@bt.error(404)
@bt.error(500)
def error_page(error):
    return bt.template("error.html", message="An error has occured", error=error)


# +------------------------------╔═══════════╗-------------------------------+ #
# |::::::::::::::::::::::::::::::║ Utilities ║:::::::::::::::::::::::::::::::| #
# +------------------------------╚═══════════╝-------------------------------+ #


def cache(ids: set[int], model: Game | Streamer) -> None:
    endpoint, cachedir = (
        ("/users", user_cache) if model is Streamer else ("/games", games_cache)
    )
    image_tag = "profile_image_url" if model is Streamer else "box_art_url"
    tmp = {i for i in ids if model.get_or_none(model.id == i) is None}
    if not tmp:
        logger.debug(f"No {endpoint[1:]} to cache")
        return None
    logger.debug(f"{len(tmp)} {endpoint[1:]} to cache")
    data = asyncio.run(AsyncRequest(endpoint).get_batch(list(tmp)))
    for datum in data:
        if model is Streamer:
            for key in ["broadcaster_type", "description", "offline_image_url"]:
                if not datum[key]:
                    datum.pop(key)
        else:
            datum["box_art_url"] = datum["box_art_url"].replace(
                "-{width}x{height}", "-285x380"
            )
    images = [Image(datum["id"], datum[image_tag]) for datum in data]

    def download_image(image: Image) -> None:
        data = httpx.get(image.url, follow_redirects=True).content
        imgpath = cachedir / f"{image.id}.jpg"
        with open(imgpath, "wb") as f:
            f.write(data)

    logger.debug(f"Downloading {len(images)} images")

    with ThreadPoolExecutor() as tp:
        tp.map(download_image, images)

    logger.debug(f"Caching {len(data)} {endpoint[1:]}")
    for datum in data:
        datum[image_tag] = f"/cache{endpoint}/{datum['id']}.jpg"
        model.create(**datum)


def time_elapsed(begin: str, d="") -> str:
    """start: UTC time"""
    start = datetime.strptime(
        begin, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    current = datetime.now(tz=timezone.utc)
    elapsed = round((current - start).total_seconds())
    delta = str(timedelta(seconds=elapsed))
    if "d" in delta:
        d = delta[: (delta.find("d") - 1)] + "d"
    h, m, _ = delta.split(" ")[-1].split(":")
    return f"{d}{h}h{m}m"


def watch_video(info: bt.FormsDict):
    c = config["USER"]
    if config.getboolean("USER", "one instance") is True:
        Media.wipe()
    if info.type == "streaming":
        command = shlex.split(
            f"streamlink -l none -p {c['media player']} -a '{c['media player args']}' {c['streamlink args']} {info.url} {c['video quality']}"
        )
    else:
        command = shlex.split(
            f"{c['media player']} {c['media player args']} --msg-level=all=no {info.url}"
        )
    Media(command, info)


def add_page(params: dict) -> dict:
    if before := bt.request.query.get("before"):
        params["before"] = before
    elif after := bt.request.query.get("after"):
        params["after"] = after
    return params


def process_vods(vods: list[dict]) -> list[Vod]:
    for vod in vods:
        vod["thumbnail_url"] = vod["thumbnail_url"].replace(
            "%{width}x%{height}", "480x270"
        )
        if not vod["thumbnail_url"]:
            vod[
                "thumbnail_url"
            ] = "https://vod-secure.twitch.tv/_404/404_processing_320x180.png"
        vod["created_at"] = time_elapsed(vod["created_at"])
    vods_fmt: list[Vod] = [
        Vod(**{k: v for k, v in vod.items() if k in Vod._fields}) for vod in vods
    ]
    return vods_fmt


def process_clips(clips: list[dict]) -> list[Clip]:
    cache({int(gid) for clip in clips if (gid := clip["game_id"])}, Game)
    for clip in clips:
        clip["time_since"] = time_elapsed(clip["created_at"])
        if clip["game_id"]:
            game: Game = Game.get(int(clip["game_id"]))
            clip["box_art_url"] = game.box_art_url
            clip["game_name"] = game.name
        if vod := clip["video_id"]:
            timestamp = time_elapsed(clip['vod_offset'])
            clip["vod_link"] = f"http://www.twitch.tv/videos/{vod}/?t={timestamp}"
        else:
            clip["vod_link"] = None
    return [Clip(**{k: v for k, v in clip.items() if k in Clip._fields}) for clip in clips]


# +------------------------╔════════════════════════╗------------------------+ #
# |::::::::::::::::::::::::║ Command Line Interface ║::::::::::::::::::::::::| #
# +------------------------╚════════════════════════╝------------------------+ #


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Web GUI for streamlink cli with Twitch account integration"
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--reset",
        action="store_true", help="reset config file")
    actions.add_argument(
        "--logout",
        action="store_true", help="remove user from app, prompt login again"
    )
    actions.add_argument(
        "-c", "--clear-data",
        action="store_true", help="remove all user data and cache"
    )
    actions.add_argument(
        "-d", "--debug",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO, help="Add debug logs to output stream"
    )
    actions.add_argument(
        "-v", "--version",
        action="version", version=f"twottle v{twottle.__version__}",
    )
    return parser


def main():
    parser = cli()
    args = parser.parse_args()
    mk_cache_dir()
    set_logger(args.loglevel)
    if args.reset:
        config.reset()
    elif args.clear_data:
        db.drop_tables([User, Streamer, Game])
        shutil.rmtree(cache_path)
        mk_cache_dir()
    elif args.logout:
        db.drop_tables([User])
    else:
        logger.info("Go to http://localhost:8080")
        try:
            waitress.serve(app=bt.app(), host="localhost",
                           threads=32, _quiet=True)
        except OSError:
            logger.info("localhost:8080 already in use")
        except KeyboardInterrupt:
            pass
        finally:
            print("\nStopping application...")
            exit()


if __name__ == "__main__":
    main()
