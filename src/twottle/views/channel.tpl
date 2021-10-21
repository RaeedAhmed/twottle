% rebase('base.tpl', title=f"{channel.display_name}")
<h2>{{channel.display_name}}</h2>
<img src="/{{channel.profile_image_url}}" alt="">
% if stream:
<div class="card">
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
      <div class="box_art"><img src="/{{stream.box_art_url}}" width="75"></div>
      <div class="text_info">
        <div class="vid_title long" title="{{stream.title}}" style="font-size: small;">{{stream.title}}</div>
        <div class="game_name long"><b>{{stream.game_name}}</b></div>
      </div>
    </div>
  </div>
% end
<br>
<a href="/c/{{channel.login}}/vods"><button class="button" style="--text-color: rgb(163, 7, 224);"><b>Vods</b></button></a>
<a href="/c/{{channel.login}}/clips"><button class="button" style="--text-color: rgb(163, 7, 224);"><b>Clips</b></button></a>