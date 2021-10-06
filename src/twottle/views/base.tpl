% setdefault('title', 'twottle')
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/static/style.css" type="text/css">
    <link href="https://css.gg/css?=|calendar-dates|eye|search|time|timer|user-add|user-remove|user" rel="stylesheet">
    <title>{{title}}</title>
    % if defined('verification'):
    <script>
        function getHash() {
            url = window.location.href.replace('#', '?')
            window.location = url
        }
    </script>
    % end
</head>

% if defined('verification'):
<body onload="getHash()">
    <h1>Caching...</h1>
</body>

% else:
<body>
    <ul>
        <li><a href="/">Home</a></li>
        <li><a href="/watching">Watching</a></li>
        <li><a href="/settings">Settings</a></li>
    </ul>
    {{!base}}
</body>
% end

</html>