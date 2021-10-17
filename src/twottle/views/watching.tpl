% rebase('base.tpl', title="Currently Watching")
% for proc in procs:
<p>{{proc.info.name}} {{proc.info.type}} {{proc.info.url}}</p>
<form action="/watching" method="POST">
    <input name="pid" value="{{proc.info.pid}}" type="hidden">
    <button>X</button>
</form>
% end
<form action="/watching" method="POST">
    <input name="wipe" value="true" type="hidden">
    <button class="button" style="--text-color: red;"><b>Close All</b></button>
</form>