

def get_object_geom_names(model, task_relevant_geom_group_name):
    return [name for name in model.geom_names if (task_relevant_geom_group_name in name)]
