% rebase('base.tpl', title="Currently Watching")
% for proc in procs:
<p>{{proc.info.type}} {{proc.info.id}} {{proc.info.url}}</p>
<form action="/watching" method="POST">
    <input name="pid" value="{{proc.info.pid}}" type="hidden">
    <button>X</button>
</form>
% end
<form action="/watching" method="POST">
    <input name="wipe" value="true" type="hidden">
    <button>Close All</button>
</form>