from .ai_navigator import AINavigatorClient


def get_default_client(*args, **kwargs):
    return AINavigatorClient(*args, **kwargs)


__all__ = ["AINavigatorClient", "get_default_client"]
