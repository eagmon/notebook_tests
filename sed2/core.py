import inspect
from typing import Callable, Any, List, Dict, Union, Optional, Tuple, Dict
import numpy as np
import json
import textwrap
from sed2 import ports


# registration
class ProcessRegistry:
    def __init__(self):
        self.registry = {}

    def register(self, process):
        if hasattr(process, '__call__'):
            self.register_function(process)

    def register_function(self, func, identifier=None):
        if not identifier:
            identifier = func.__name__
        signature = inspect.signature(func)
        annotation = getattr(func, 'annotation', None)
        ports = getattr(func, 'ports')

        # TODO -- assert ports and signature match
        if not annotation:
            raise Exception(f'Process {identifier} requires annotations')
        if not ports:
            raise Exception(f'Process {identifier} requires annotations')

        item = {
            'annotation': annotation,
            'ports': ports,
            'address': func}
        self.registry[identifier] = item

    def access(self, name):
        return self.registry.get(name)

    def get_annotations(self):
        return [v.get('annotation') for k, v in self.registry.items()]

    def activate_process(self, process_name, namespace):
        namespace[process_name] = self.registry[process_name]['address']

    def activate_all(self, namespace):
        """how to add to globals: process_registry.activate_all(globals())"""
        for process_name in self.registry.keys():
            self.activate_process(process_name, namespace)


def register_functions(functions, process_registry=None):
    if not process_registry:
        process_registry = ProcessRegistry()
    for func in functions:
        process_registry.register(func)
    return process_registry


def serialize_instance(wiring):
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type "{obj.__class__.__name__}" is not JSON serializable')

    return json.dumps(wiring, default=convert_numpy)


def deserialize_instance(serialized_wiring):
    if isinstance(serialized_wiring, dict):
        return serialized_wiring

    def convert_numpy(obj):
        if isinstance(obj, list):
            return np.array(obj)
        return obj

    return json.loads(serialized_wiring, object_hook=convert_numpy)


def topological_sort(graph):
    """Return list of sorted process names based on dependencies"""
    visited = set()
    sorted_list = []

    def visit(node):
        if node not in visited:
            visited.add(node)
            if node in graph:
                if '_depends_on' in graph[node]:
                    for neighbor in graph[node]['_depends_on']:
                        visit(neighbor)
            sorted_list.append(node)

    for node in graph:
        visit(node)
    return sorted_list


def get_processes_and_states_from_schema(schema, process_registry):
    all_annotations = process_registry.get_annotations()

    # separate the processes and states
    processes = {}
    states = {}
    for name, value in schema.items():
        if isinstance(value, dict) and value.get('wires'):
            annotation = value.get('_type')
            if annotation not in all_annotations:
                raise Exception(
                    f'{name} has a type annotation {annotation} not included in the process registry')
            processes[name] = value
        else:
            states[name] = value
    return processes, states


def generate_script(
        schema: Union[str, dict],
        process_registry: ProcessRegistry
) -> str:
    """Generate an executable Python script from the declarative JSON format"""

    schema = deserialize_instance(schema)
    script = []

    # separate the processes and states
    processes, states = get_processes_and_states_from_schema(schema, process_registry)
    sorted_processes = topological_sort(processes)

    # add states to the top of the script
    for name, value in states.items():
        script.insert(0, f'{name} = {value}')

    # add processes to the bottom of the script in their sorted order
    for name in sorted_processes:
        value = processes[name]
        process_entry = process_registry.access(name)
        if process_entry:
            ports = process_entry['ports']
            wires = value['wires']
            func_script = None
            inputs = ports.get('inputs')
            outputs = ports.get('outputs')
            if inputs:
                input_values = [(key, wires[key]) for key in inputs.keys()]
                input_args = ', '.join(f'{arg}={val}' for (arg, val) in input_values)
                func_script = f'{name}({input_args})'
            else:
                func_script = f'{name}()'
            if outputs:
                output_values = [(key, wires[key]) for key in outputs.keys()]
                output_args = ', '.join(f'{val}' for (arg, val) in output_values)
                func_script = f'{output_args} = ' + func_script
            script.append(func_script)
        else:
            raise Exception(f'Function {name} not found in the process registry.')
    return '\n'.join(script)


def generate_composite_process(json_str, process_registry):
    deserialized_wiring = deserialize_instance(json_str)
    ports = deserialized_wiring.pop('_ports')
    input_ports = ports.get('inputs')
    output_ports = ports.get('outputs')
    input_port_values = {}
    for port_name in input_ports.keys():
        input_port_values[port_name] = deserialized_wiring.pop(port_name, None)
    script = generate_script(deserialized_wiring, process_registry)

    # activate all processes
    process_registry.activate_all(globals())

    # make the composite process as a string
    func_str = ''
    return_str = ''
    if ports:
        func_str += f'@ports({ports})\n'
    if input_ports:
        input_args_str = ', '.join(
            [f'{port_name}: {value} = {input_port_values[port_name]}' for port_name, value in input_ports.items()])
        func_str += f'def composite_process({input_args_str})'
    else:
        func_str += f'def composite_process()'
    if output_ports:
        output_args_str = ', '.join([f'{value}' for key, value in output_ports.items()])
        func_str += f' -> {output_args_str}:\n'
        return_str = ', '.join([f'{key}' for key, value in output_ports.items()])
        return_str = '\nreturn ' + return_str
    else:
        func_str += f':\n'

    indent = '    '
    indented_script = textwrap.indent(script, indent)
    indented_return_str = textwrap.indent(return_str, indent)
    func_str += indented_script
    func_str += indented_return_str

    print(func_str)
    exec(func_str, globals())
    composite_process = globals()['composite_process']
    return composite_process


def get_process_schema(process):
    ports = process.ports
    input_ports = ports.get('inputs', {})
    output_ports = ports.get('outputs', {})
    name = process.__name__
    schema = {name: {
        '_ports': {**input_ports, **output_ports}}}
    return schema
