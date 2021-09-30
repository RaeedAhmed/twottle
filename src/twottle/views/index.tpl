% rebase('base.tpl', title="twitch-py")
<main>
    <h1>{{user.login}}</h1>
    % for stream in streams:
        <article>
            <h3><img src="{{stream.profile_image_url}}" width="75"> {{stream.user_name}}</h3>
            <p>{{stream.title}}</p>
            <a href="/watch?type=stream&id={{stream.user_id}}&url=twitch.tv/{{stream.user_login}}"><img src="{{stream.thumbnail_url}}" alt="" width="240"></a>
            <p>{{stream.viewer_count}}</p>
            <div>
                <p>{{stream.game_name}}</p>
                <img src="{{stream.box_art_url}}" alt="" width=50>
            </div>
        </article>
    % end
</main>



