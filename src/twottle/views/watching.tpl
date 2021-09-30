% rebase('base.tpl', title="Currently Watching")
% for proc in procs:
<p>{{proc.info.type}} {{proc.info.id}} {{proc.info.url}}</p>
% end