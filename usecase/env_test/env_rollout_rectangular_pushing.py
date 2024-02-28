from domain_object_builder import DomainObject
from domain_object_builder import DomainObjectBuilder
from domain_object_director import CheckEnvRolloutDirector
from service import NTD, create_interpolated_matrix_by_divisions
from service import create_gif_from_Image_list
import numpy as np
from usecase.xml_generation.rectangular_object_xml_generation import rectangular_object_xml_generation

class CheckEnvDataset:
    @staticmethod
    def run(domain_object: DomainObject, mass_list: list):
        env                  = domain_object.env
        init_state           = domain_object.init_state
        TaskSpaceValueObject = domain_object.TaskSpaceValueObject
        config_xml           = domain_object.config_xml_generation
        # ---
        xml_path        = domain_object.original_xml_path
        task_space_ctrl = init_state["task_space_position"].value
        # ---
        # num_epoch           = 1
        num_task_step       = 30
        task_space_ctrl_seq = create_interpolated_matrix_by_divisions(
            task_space_ctrl.squeeze(), divisions=num_task_step, dims=[0, 1])
        # ---

        # ---
        for i in range(len(mass_list)):
            rectangular_object_xml_generation(config_xml, mass=mass_list[i])
            # ---
            env.set_xml_path(xml_path)
            env.load_model()
            env.reset(init_state)
            env.render()
            # ---
            img = []
            for t in range(num_task_step):
                # print("step t = {}".format(t))
                render = env.render()    # ; print(render)
                state  = env.get_state() # ; print(state)
                env.view()
                # import ipdb; ipdb.set_trace()
                # ---
                env.set_task_space_ctrl(TaskSpaceValueObject(NTD(task_space_ctrl_seq[t])))
                env.step(is_view=True)
                # ---
                # import ipdb; ipdb.set_trace()
                # img.append(render["canonical"].value)
            # ---
            # create_gif_from_Image_list(
            #     images   = img,
            #     fname    = "./rectangular_push_[mass={}].gif".format(config_xml.object_type.mass),
            #     duration = 250,
            # )


if __name__ == '__main__':
    # ---
    # env_name = "sim/pushing"
    env_name = "sim/rectangular_pushing"
    # ---
    # mass_list = [
    #     [0.1,  0.02, 0.02], # 1
    #     [0.1,  0.1,  0.02], # 2
    #     [0.1,  0.1,  0.1] , # 6
    #     [0.02, 0.1,  0.1] , # 4
    #     [0.02, 0.02, 0.1] , # 3
    #     [0.02, 0.02, 0.02], # 5
    #     [0.02, 0.1,  0.02], # 7
    # ]
    mass_list = [
        [0.1,  0.01, 0.01, 0.01], # 1
        [0.1,  0.1,  0.01, 0.01], # 1
        [0.1,  0.1,  0.1,  0.01], # 1
        [0.1,  0.1,  0.1,  0.1], # 1
        [0.01, 0.1,  0.1,  0.1], # 1
        [0.01, 0.01, 0.1,  0.1], # 1
        [0.01, 0.01, 0.01, 0.1], # 1
        # ----
        [0.1,  0.01, 0.01, 0.01], # 1
        [0.1,  0.1,  0.01, 0.01], # 1
        [0.1,  0.1,  0.1,  0.01], # 1
        [0.1,  0.1,  0.1,  0.1], # 1
        [0.01, 0.1,  0.1,  0.1], # 1
        [0.01, 0.01, 0.1,  0.1], # 1
        [0.01, 0.01, 0.01, 0.1], # 1
        # ----
        [0.1,  0.01, 0.01, 0.01], # 1
        [0.1,  0.1,  0.01, 0.01], # 1
        [0.1,  0.1,  0.1,  0.01], # 1
        [0.1,  0.1,  0.1,  0.1], # 1
        [0.01, 0.1,  0.1,  0.1], # 1
        [0.01, 0.01, 0.1,  0.1], # 1
        [0.01, 0.01, 0.01, 0.1], # 1
    ]
    # ---
    builder       = DomainObjectBuilder()
    director      = CheckEnvRolloutDirector()
    domain_object = director.construct(builder, env_name=env_name)
    CheckEnvDataset.run(domain_object, mass_list)
