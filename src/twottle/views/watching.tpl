% rebase('base.tpl', title="Currently Watching")
% for proc in procs:
<form action="/watching" method="POST">
    <label for="pid">{{proc.info.name}} {{proc.info.type}} {{proc.info.url}}</label>
    <input name="pid" value="{{proc.info.pid}}" type="hidden">
    <button class="button" style="--text-color: red; padding: 5px 10px;"><b>X</b></button>
</form>

% end
<br>
<form action="/watching" method="POST">
    <input name="wipe" value="true" type="hidden">
    <button class="button" style="--text-color: red;"><b>Close All</b></button>
</form>