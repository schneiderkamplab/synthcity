# stdlib
import glob
from os.path import basename, dirname, isfile, join

def_categories = [
    "generic",
    "privacy",
    "survival_analysis",
    "time_series",
    "domain_adaptation",
    "images",
    "debug",
]
plugins = {}

for cat in def_categories:
    plugins[cat] = glob.glob(join(dirname(__file__), cat, "plugin*.py"))


class Plugins:
    def __init__(self, categories: list = def_categories) -> None:
        from synthcity.plugins.core.plugin import Plugin, PluginLoader

        plugins_to_use = []
        for cat in categories:
            plugins_to_use.extend(plugins[cat])

        self._loader = PluginLoader(plugins_to_use, Plugin, categories)

    def __getattr__(self, name: str):
        return getattr(self._loader, name)


def __getattr__(name: str):
    if name in {"Plugin", "PluginLoader"}:
        from synthcity.plugins.core.plugin import Plugin, PluginLoader

        return {"Plugin": Plugin, "PluginLoader": PluginLoader}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [basename(f)[:-3] for f in plugins[cat] for cat in plugins if isfile(f)] + [
    "Plugins",
    "Plugin",
]
