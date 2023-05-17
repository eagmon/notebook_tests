import itertools
import numpy as np
import matplotlib.pyplot as plt
import tellurium as te
import pandas as pd
from sed1.core import register_functions
from sed1 import ports, annotate, register


class Model:
    def __init__(
            self,
            sedml_language: str,
            sbml_file: str
    ):
        self.sedml_language = sedml_language
        self.sbml_file = sbml_file
        self.sbml_model = te.loadSBMLModel(self.sbml_file)

    def set(
            self,
            element_id: str,
            value: float
    ):
        if element_id in self.sbml_model.getFloatingSpeciesIds():
            self.sbml_model.setValue(element_id, value)
        elif element_id in self.sbml_model.getGlobalParameterIds():
            self.sbml_model.setValue(element_id, value)
        else:
            raise Exception(f'species {element_id} does not exist')

    def reset(self):
        self.sbml_model.reset()

    def get_value(self, element_id: str) -> float:
        return self.sbml_model.getValue(element_id)


@ports({'inputs': {
    'path_to_sbml': 'str'},
    'outputs': {
        'model': 'Model'}})
@annotate('sed:sbml_model_from_path')
def sbml_model_from_path(path_to_sbml):
    return Model('urn:sedml:language:sbml', path_to_sbml)


@ports({'inputs': {'model_instance': 'Model'}})
@annotate('sed:model_reset')
def model_reset(model_instance):
    model_instance.reset()


@ports({
    'inputs': {
        'model_instance': 'Model',
        'element_id': 'str',
        'value': 'float'}})
@annotate('sed:set_model')
def model_set_value(model_instance: Model, element_id, value):
    model_instance.set(element_id, value)


@ports({
    'inputs': {
        'model_instance': 'Model',
        'element_id': 'str'},
    'outputs': {
        'value': 'float'}})
@annotate('sed:get_model')
def model_get_value(model_instance, element_id):
    return model_instance.get_value(element_id)


@ports({
    'inputs': {
        'model': 'Model',
        'time_start': 'float',
        'time_end': 'float',
        'num_points': 'int',
        'selection_list': 'List[str]'},
    'outputs': {
        'results': 'dict'}})
@annotate('sed:uniform_time_course')
def uniform_time_course(
        model,
        time_start,
        time_end,
        num_points,
        selection_list=None
):
    sim_result = model.sbml_model.simulate(start=time_start, end=time_end, points=num_points, selections=selection_list)
    return {column: sim_result[column] for column in sim_result.colnames}


@ports({
    'inputs': {
        'data_file': 'str',
        'file_format': 'str'},
    'outputs': {
        'data': 'pd.DataFrame'}})
@annotate('sed:data_description')
def data_description(data_file, file_format):
    if file_format == 'CSV':
        data = pd.read_csv(data_file)
    return data


@ports({
    'inputs': {
        'sim': 'Dict[str, np.ndarray]',
        'data': 'pd.DataFrame'},
    'outputs': {
        'sum': 'float'}})
@annotate('sed:sum_of_squares')
def sum_of_squares(sim, data):
    sim_df = pd.DataFrame(sim)
    diff = sim_df.set_index('time') - data.set_index('time')
    return np.sum(diff ** 2).sum()


class curve:
    def __init__(
            self,
            x_values: np.ndarray,
            y_values: np.ndarray,
            name=None
    ):
        self.x_values = x_values
        self.y_values = y_values
        self.name = name


@ports({
    'inputs': {
        'model': 'Model',
        'selection_list': 'list'},
    'outputs': {
        'results': 'dict'}})
@annotate('sed:steady_state_values')
def steady_state_values(model, selection_list):
    sbml_model = te.loadSBMLModel(model.sbml_file)
    steady_state_values = sbml_model.getSteadyStateValues()
    ids = sbml_model.getFloatingSpeciesIds() + sbml_model.getGlobalParameterIds()
    result_dict = {}
    for i, id in enumerate(ids):
        if id in selection_list:
            result_dict[id] = steady_state_values[i]
    return result_dict


@ports({
    'inputs': {
        'model': 'Model',
        'tolerance': 'float'},
    'outputs': {
        'sum': 'float'}})
@annotate('sed:is_steady_state')
def is_steady_state(model, tolerance):
    sbml_model = te.loadSBMLModel(model.sbml_file)
    ss = sbml_model.steadyState()
    return ss < tolerance


@ports({
    'inputs': {
        'model': 'Model',
        'variable': 'string',
        'parameter': 'string'},
    'outputs': {
        'coefficient': 'float'}})
@annotate('sed:is_steady_state')
def concentration_control_coefficient(model, variable, parameter):
    sbml_model = te.loadSBMLModel(model.sbml_file)
    return sbml_model.getCC(variable, parameter)


def report_dict(result):
    for key, value in result.items():
        print(f'{key}: {value}')


@ports({
    'inputs': {
        'results': 'Any',
        'title': 'str'}})
@annotate('sed:report')
def report(results, title):
    if title:
        print(title)
    if isinstance(results, list):
        for result in results:
            report_dict(result)
    if isinstance(results, dict):
        report_dict(results)


@ports({
    'inputs': {
        'input_dict': 'dict',
        'model': 'Model',
        'data_description': 'pd.DataFrame',
        'time_start': 'float',
        'time_end': 'float',
        'num_points': 'int'},
    'outputs': {
        'results': 'dict'}})
@annotate('sed:n_dimensional_scan')
def n_dimensional_scan(
        input_dict,
        model,
        data_description,
        time_start,
        time_end,
        num_points,
):
    combinations = list(itertools.product(*input_dict.values()))
    keys = list(input_dict.keys())
    selection_list = keys + ['time']
    results = {}
    for c in combinations:
        model.reset()
        for idx, v in enumerate(c):
            model.set(keys[idx], v)
        sim1 = uniform_time_course(
            model, time_start, time_end, num_points, selection_list=selection_list)
        results[c] = sum_of_squares(sim1, data_description)
    return results


@ports({
    'inputs': {'model_instance': 'Model', 'config': 'dict'},
    'outputs': {'results': 'list'}})
@annotate('sed:repeated_simulation')
def repeated_simulation(model_instance, config):
    results = []
    for key, values in config.items():
        for value in values:
            model_instance.reset()
            model_instance.set(key, float(value))
            sim1 = uniform_time_course(model_instance, 0, 10, 5, selection_list=['time', 'S',
                                                                                 'Z'])  # TODO -- selection list should be configurable!
            results.append(sim1)
    return results


@ports({
    'inputs': {
        'results': 'dict',
        'curves': 'dict',
        'name': 'str'}})
@annotate('sed:plot2d')
def plot2d(results, curves, name):
    plt.figure(name)
    for curve_name, axes in curves.items():
        x_values = results[axes['x']]
        y_values = results[axes['y']]
        curve_obj = curve(x_values, y_values, name=curve_name)
        plt.plot(curve_obj.x_values, curve_obj.y_values, label=curve_obj.name)
    plt.legend()
    plt.show()


# register functions
functions = [
    sbml_model_from_path,
    uniform_time_course,
    model_reset,
    model_set_value,
    model_get_value,
    data_description,
    sum_of_squares,
    plot2d,
    steady_state_values,
    is_steady_state,
    report,
    n_dimensional_scan,
    repeated_simulation,
    concentration_control_coefficient,
]

sed_process_registry = register_functions(functions)
