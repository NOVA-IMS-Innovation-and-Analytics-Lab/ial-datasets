from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_process_funcs


def create_pipeline(**kwargs) -> Pipeline:
    process_funcs = generate_process_funcs()
    nodes = []
    for data_name, factor, process_func in process_funcs:
        input_data_name = (
            f'{data_name}_numerical_features_binary_target_imbalanced_data'
        )
        output_data_name = (
            f'{data_name}_numerical_features_binary_target_imbalanced_data_{factor}'
        )
        nodes.append(
            node(
                func=process_func,
                inputs=[input_data_name, 'parameters'],
                outputs=output_data_name,
                name=f'{data_name}_{factor}_node',
            )
        )
    return pipeline(nodes)
