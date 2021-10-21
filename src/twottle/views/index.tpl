% rebase('base.tpl', title="twottle")
<main>
    <h1>{{user.display_name}}</h1>
    <div class="card-grid">
        % for stream in streams:
        <div class="card">
          <div class="card-top">
            <div class="profile_pic"><a href="/c/{{stream.user_login}}"><img src="{{stream.profile_image_url}}" width="100%"></a></div>
            <div class="display_name"><h3>{{stream.user_name}}</h3></div>
            <div class="channel_links" style="display: inline-flex; position: relative;">
              <a href="javascript: void(0)" onclick="popup('https://www.twitch.tv/popout/{{stream.user_login}}/chat?popout=')">
                <i class="gg-comment"></i>
              </a>
              <a href="https://twitch.tv/{{stream.user_login}}/about" target="_blank"><i class="gg-arrow-top-right-r"></i></a>
            </div>
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
                    <div class="c" style="--ggs: 5">
                        <i class="gg-play-button"></i>
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



