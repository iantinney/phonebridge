# Contributing

Thanks for helping. The most useful things right now:

1. **A first-task report** from your Mac and phone, whether it worked or not ([the issue form](.github/ISSUE_TEMPLATE/first-task.yml)).
2. **Known-screen records** for app screens Phonebridge doesn't recognise yet.
3. **Bug fixes with a test.**

## Develop

```sh
uv sync
uv run pytest                            # the bridge; Linux or macOS, no phone needed
uv run pytest packages/workspace/tests   # the desktop workspace, against a simulated bridge
```

The tests use synthetic frames and fake backends. Anything that needs a phone is a manual check: say in the pull
request what you ran (`phonebridge check <alias>`, a prompt) on which device and iOS version.

## Known-screen records

Records live in `src/phonebridge/screens/records/<platform>/`, one YAML file per screen (`screens/schema.py` defines
the fields), with match templates in `records/templates/`. `uv run phonebridge screens list` and
`uv run phonebridge screens gates` check them against the corpus.

A record describes a screen by its layout and fixed labels. Never commit text that identifies a person: no handles,
names, message text, profile pictures or feed content from real accounts, yours included. Use made-up examples in
notes, and crop templates to the app's own icons.

## Recipes and writes

A recipe is a task anyone can rerun on new inputs. Propose one with the [recipe form](.github/ISSUE_TEMPLATE/recipe.yml)
after running it on a real phone. Read-only recipes come first. Anything that posts, sends, follows, likes or edits
must ask a person to approve each action, and must stop on a challenge, a warning or an unfamiliar screen. We don't
accept code that tries to make an app unable to tell it is automated, or that fakes engagement.

## Pull requests

- Keep each change focused, with a test where the code can be tested off-device.
- Never commit device UDIDs (use `00008101-XXXXXXXXXXXXXXXX`), tokens, screenshots of real accounts or recorded
  sessions.
- Phonebridge is GPL-3.0-or-later. By contributing you agree that your contribution is licensed the same way.
