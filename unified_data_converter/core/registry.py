import functools

class ProcessorRegistry:
    """
    Registry for data processing functions.
    Processors must have the signature: func(target_path, params, context)
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._registry[name] = func
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

    @classmethod
    def list_processors(cls):
        return list(cls._registry.keys())

