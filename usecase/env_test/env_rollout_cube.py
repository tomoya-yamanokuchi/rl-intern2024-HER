from domain_object_builder import DomainObject
from domain_object_builder import DomainObjectBuilder
from domain_object_director import CheckEnvRolloutDirector
from service import NTD, ListedImageSaverAsGif
import numpy as np



class CheckEnvDataset:
    @staticmethod
    def run(domain_object: DomainObject):
        env                      = domain_object.env
        init_state               = domain_object.init_state
        TaskSpaceValueObject     = domain_object.TaskSpaceValueObject
        TaskSpaceDiffValueObject = domain_object.TaskSpaceDiffValueObject
        xml_path                 = domain_object.original_xml_path
        # ---
        task_space_position_init = init_state["task_space_position"]


        for i in range(3):
            # import ipdb; ipdb.set_trace()
            env.set_xml_path(xml_path)
            env.load_model()
            env.reset(init_state)

            # for k in range(5):
            # import ipdb; ipdb.set_trace()
            env.set_task_space_ctrl(task_space_position_init)
            env.step(is_view=True)

            env.render()
            # ---
            image_saver = ListedImageSaverAsGif()
            image_saver.initialize_container()
            # ---
            # import ipdb; ipdb.set_trace()
            for t in range(20):
                # render = env.render()    # ; print(render)
                state  = env.get_state() # ; print(state)

                # import ipdb; ipdb.set_trace()
                print("object_rotation = [{:.3f}, {:.3f}, {:.3f}]".format(
                    *state["object_rotation"].value))

                env.view()
                # ---
                # ud                   = np.zeros_like(task_space_position_init.value)
                # import ipdb; ipdb.set_trace()
                ud                   = np.random.randn(*task_space_position_init.value.shape)*0.2
                # ---
                task_space_diff_ctrl = TaskSpaceDiffValueObject(ud)
                task_space_ctrl      = state["task_space_position"] + task_space_diff_ctrl
                ctrl                 = env.set_task_space_ctrl(task_space_ctrl)
                # print("step t = {} : ctrl = {}".format(t, task_space_ctrl.value.squeeze()))
                # ----
                env.step(is_view=True)
                # import ipdb; ipdb.set_trace()
                # ---
                # image_saver.append(render["canonical"].value)

            # image_saver.save_as_gif(save_path="./env_cube_camera_pattern{}.gif".format(i), duration=100)


if __name__ == '__main__':

    builder       = DomainObjectBuilder()
    director      = CheckEnvRolloutDirector()
    domain_object = director.construct(builder, env_name="sim/cube")
    CheckEnvDataset.run(domain_object)
