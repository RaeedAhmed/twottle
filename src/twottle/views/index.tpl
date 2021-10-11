% rebase('base.tpl', title="twottle")
<main>
    <h1>{{user.display_name}}</h1>
    % for stream in streams:
        <article class="card">
            <h3><img src="{{stream.profile_image_url}}" width="75"> {{stream.user_name}}</h3>
            <p>{{stream.title}}</p>
            <div class="thumbnail">
                <a href="/watch?type=stream&id={{stream.user_id}}&url=twitch.tv/{{stream.user_login}}"><img src="{{stream.thumbnail_url}}"></a>
                <div class="bl">
                    <i class="gg-time"></i>
                    <b>{{stream.uptime}}</b>
                </div>
                <div class="br">
                    <i class="gg-user" style="color: #ff496c;"></i>
                    <b>{{stream.viewer_count}}</b>   
                </div>
            </div>
            <div>
                <p><img src="{{stream.box_art_url}}" alt="" width=50> {{stream.game_name}}</p>       
            </div>
        </article>
    % end
</main>



