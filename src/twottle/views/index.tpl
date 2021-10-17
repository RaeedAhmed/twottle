% rebase('base.tpl', title="twottle")
<main>
    <h1>{{user.display_name}}</h1>
    <div class="card-grid">
        % for stream in streams:
        <div class="card">
          <div class="card-top">
            <div class="profile_pic"><img src="{{stream.profile_image_url}}" width="100%"></div>
            <div class="display_name"><h3>{{stream.user_name}}</h3></div>
          </div>
          <div class="card-mid">
                <div class="thumbnail">
                    <a href="/watch?type=streaming&name={{stream.user_name}}&url=twitch.tv/{{stream.user_login}}"><img src="{{stream.thumbnail_url}}"></a>
                    <div class="bl">
                        <i class="gg-time"></i>
                        <b>{{stream.uptime}}</b>
                    </div>
                    <div class="br">
                        <i class="gg-user" style="color: #ff496c;"></i>
                        <b>{{stream.viewer_count}}</b>   
                    </div>
                </div>
          </div>
          <div class="card-bottom">
            <div class="box_art"><img src="{{stream.box_art_url}}" width="75"></div>
            <div class="text_info">
              <div class="vid_title long" title="{{stream.title}}" style="font-size: small;">{{stream.title}}</div>
              <div class="game_name long"><b>{{stream.game_name}}</b></div>
            </div>
          </div>
        </div>
        % end
    </div>
</main>



