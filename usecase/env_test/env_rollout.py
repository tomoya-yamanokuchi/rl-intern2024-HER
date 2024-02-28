from domain_object_builder import DomainObject
from domain_object_builder import DomainObjectBuilder
from domain_object_director import CheckEnvRolloutDirector
from service import NTD
import numpy as np


class CheckEnvDataset:
    @staticmethod
    def run(domain_object: DomainObject):
        env             = domain_object.env
        init_state      = domain_object.init_state
        TaskSpaceDiff   = domain_object.TaskSpaceDiffValueObject
        # ---
        xml_path        = domain_object.original_xml_path
        task_space_ctrl = init_state["task_space_position"].value
        # ---
        num_task_step   = 30
        task_space_ctrl = np.linspace(0, 1, num_task_step)
        task_space_ctrl = np.concatenate([
            task_space_ctrl.reshape(-1, 1),
            task_space_ctrl.reshape(-1, 1),
            task_space_ctrl.reshape(-1, 1)],
            axis = -1
        )
        import ipdb; ipdb.set_trace()
        # ---
        for i in range(5):
            env.set_xml_path(xml_path)
            env.reset(init_state)
            env.render()
            # import ipdb; ipdb.set_trace()
            for t in range(num_task_step):
                print("step t = {}".format(t))
                # render = env.render()    ; print(render)
                state  = env.get_state() ; print(state)
                env.view()
                # ---
                # task_space_ctrl += 0.05
                # ---
                env.set_task_space_ctrl(TaskSpaceDiff(NTD(task_space_ctrl[t])))
                env.step(is_view=True)
                # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    # ---
    env_name = "sim/pushing"
    # ---
    builder       = DomainObjectBuilder()
    director      = CheckEnvRolloutDirector()
    domain_object = director.construct(builder, env_name=env_name)
    CheckEnvDataset.run(domain_object)
