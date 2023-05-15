"""
Decorators for Processes
"""


def register(registry):
    def decorator(func):
        registry.register_functions(func)
        return func
    return decorator


def annotate(annotation):
    def decorator(func):
        func.annotation = annotation
        return func
    return decorator


def ports(ports_schema):
    # assert inputs/outputs and types, give suggestions
    allowable = ['inputs', 'outputs']
    assert all(key in allowable for key in
               ports_schema.keys()), f'{[key for key in ports_schema.keys() if key not in allowable]} not allowed as top-level port keys. Allowable keys include {str(allowable)}'

    # TODO assert type are in type_registry
    # TODO check that keys match function signature
    def decorator(func):
        func.ports = ports_schema
        return func

    return decorator
