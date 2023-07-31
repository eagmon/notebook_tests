import os
from graphviz import Source
import re


def edit_graphviz(input_file_path, new_attributes, output_file_path):
    # Read the original graphviz content from the file
    with open(input_file_path, 'r') as f:
        original_graphviz = f.read()

    # Find the patterns to be replaced
    graph_pattern = r'(dpi=.* size=".*")|(overlap=false rankdir=.*)'
    # node_pattern = r'node \[.*\]'

    # Find the new attributes for graph and node
    new_graph_attributes = re.search(graph_pattern, new_attributes).group(0)
    # new_node_attributes = re.search(node_pattern, new_attributes).group(0)

    # Replace the patterns with the new attributes
    updated_graphviz = re.sub(graph_pattern, new_graph_attributes, original_graphviz)
    # updated_graphviz = re.sub(node_pattern, new_node_attributes, updated_graphviz)

    # Write the updated graphviz content to the new file
    with open(output_file_path, 'w') as f:
        f.write(updated_graphviz)

    print(f"Updated graphviz content written to: {output_file_path}")


def get_absolute_path(relative_path):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_dir, relative_path)


def generate_graph_from_file(file_path, output_file=None):
    # Read the content of the file
    with open(file_path, 'r') as file:
        graph_data = file.read()

    # Create a Source object from the Graphviz data
    graph = Source(graph_data)

    # Render the graph to a file (default format is PDF, so we explicitly specify PNG)
    graph.render(filename=output_file, format='png', cleanup=True)


def create_modified_graph():
    # input_file_path = 'out/NIH_NIA/gut_microbiome'
    modified_file_path = 'out/NIH_NIA/gut_microbiome_modified'
    output_graph_file = 'out/NIH_NIA/gut_microbiome_graph'
    # new_attributes = '''
    # graph [nodesep=0.02, ranksep=0.02 dpi=300 overlap=false size="4,8" ratio="fill"]
    # '''
    # # make a new graphvi file
    # edit_graphviz(
    #     input_file_path=input_file_path,
    #     new_attributes=new_attributes,
    #     output_file_path=modified_file_path)

    # plot it and save
    file_path = get_absolute_path(modified_file_path)
    generate_graph_from_file(file_path=file_path, output_file=output_graph_file)


if __name__ == '__main__':
    create_modified_graph()
