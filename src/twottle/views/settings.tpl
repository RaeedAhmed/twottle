% rebase('base.tpl', title="Settings")

<form action="/settings" method="POST">
    % for setting, value in sorted(list(user.items())):
    <div>
        <label for="{{setting}}">{{setting}}</label>
        <input value="{{value}}" name="{{setting}}" id="{{setting}}">
    </div>
    % end
    <div>
        <button>Apply Settings</button>
    </div>
</form>

<form action="/settings" method="POST">
    <input name="reset" id="reset" type="submit" value="Restore Defaults">
</form>

% for change in changes:
<p>{{change}}</p>
% end
