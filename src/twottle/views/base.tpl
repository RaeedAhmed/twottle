% setdefault('title', 'twottle')
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/static/style.css" type="text/css">
    <link href="https://css.gg/css?=|calendar-dates|eye|search|time|timer|user-add|user-remove|user|play-button|comment|arrow-top-right-r" rel="stylesheet">
    <title>{{title}}</title>
    % if defined('verification'):
    <script>
        function getHash() {
            url = window.location.href.replace('#', '?')
            window.location = url
        }
    </script>
    % end
    <script type="text/javascript">
        function popup(url) 
        {
         var width  = 400;
         var height = screen.availHeight;
         var left   = screen.availWidth - width;
         var top    = 0;
         var params = 'width='+width+', height='+height;
         params += ', top='+top+', left='+left;
         params += ', directories=no';
         params += ', location=no';
         params += ', menubar=no';
         params += ', resizable=no';
         params += ', scrollbars=no';
         params += ', status=no';
         params += ', toolbar=no';
         newwin=window.open(url,'customWindow', params);
         if (window.focus) {newwin.focus()}
         return false;
        }
        </script>
</head>

% if defined('verification'):
<body onload="getHash()">
    <h1>Caching...</h1>
</body>

% else:
<body>
    <div class="page_layout">
        <div class="navbar">
            <div class="topnav">
                <a class="hyperlink" href="/">Home</a>
                <a class="hyperlink" href="/watching">Watching</a>
                <a class="hyperlink" href="/settings">Settings</a>
            </div>
        </div>
        <div class="page_content">
            {{!base}}
        </div>
        <div class="footer">
            <div class="centered">
                <a href="https://github.com/raeedahmed/twottle"><b>GitHub</b></a>
            </div>              
        </div>
    </div>
</body>
% end




</html>
