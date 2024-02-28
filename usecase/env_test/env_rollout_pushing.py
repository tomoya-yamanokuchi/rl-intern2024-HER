from domain_object_builder import DomainObject
from domain_object_builder import DomainObjectBuilder
from domain_object_director import CheckEnvRolloutDirector
from service import NTD
import numpy as np


class CheckEnvDataset:
    @staticmethod
    def run(domain_object: DomainObject):
        env                      = domain_object.env
        init_state               = domain_object.init_state
        TaskSpaceValueObject     = domain_object.TaskSpaceValueObject
        TaskSpaceDiffValueObject = domain_object.TaskSpaceDiffValueObject
        # ---
        task_space_position_init = init_state["task_space_position"]

        for i in range(5):
            env.reset(init_state)

            # for k in range(5):
            env.set_task_space_ctrl(task_space_position_init)
            env.step(is_view=True)

            env.render()
            # import ipdb; ipdb.set_trace()
            for t in range(5):
                render = env.render()    # ; print(render)
                state  = env.get_state() # ; print(state)
                env.view()
                # ---
                ud                   = np.zeros_like(task_space_position_init.value)
                ud[:, :, 1:2:2]     += 0.15
                task_space_diff_ctrl = TaskSpaceDiffValueObject(ud)
                task_space_ctrl      = state["task_space_position"] + task_space_diff_ctrl
                ctrl                 = env.set_task_space_ctrl(task_space_ctrl)
                print("step t = {} : ctrl = {}".format(t, task_space_ctrl.value.squeeze()))
                # ----
                env.step(is_view=True)
                # import ipdb; ipdb.set_trace()

if __name__ == '__main__':

    builder       = DomainObjectBuilder()
    director      = CheckEnvRolloutDirector()
    domain_object = director.construct(builder, env_name="sim/pushing")
    CheckEnvDataset.run(domain_object)
