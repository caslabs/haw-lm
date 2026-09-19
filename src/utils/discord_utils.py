import os
from discord_webhook import DiscordWebhook

def discord_notify(lines):
    """Send a list of strings as one message to Discord (if webhook set)."""
    hook = os.getenv("DISCORD_HOOK")
    if hook:
        DiscordWebhook(hook, content="\n".join(lines)).execute()
