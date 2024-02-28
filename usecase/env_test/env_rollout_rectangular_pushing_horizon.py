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
        num_task_step       = 6
        task_space_ctrl_seq = create_interpolated_matrix_by_divisions(
            task_space_ctrl.squeeze(), divisions=num_task_step, dims=[0, 1])
        # ---
        x_traj = np.linspace(start=0.78, stop=0.70, num=7, endpoint=True) [1:]
        y_traj = np.linspace(start=0.49, stop=0.63, num=7, endpoint=True) [1:]
        # ---
        task_space_ctrl_seq[:6, 0] = x_traj
        task_space_ctrl_seq[:6, 1] = y_traj
        # import ipdb; ipdb.set_trace()
        # task_space_ctrl_seq = np.stack([x_traj, y_traj], axis=-1)
        num_task_step = task_space_ctrl_seq.shape[0]
        # ----
        # theta   = np.pi / 4. # 元の座標空間からの回転角度
        # X       = task_space_ctrl_seq[:, 0]
        # Y       = task_space_ctrl_seq[:, 1]
        # X_prime = X * np.cos(theta) - Y * np.sin(theta)
        # Y_prime = X * np.sin(theta) + Y * np.cos(theta)
        # # ---
        # task_space_ctrl_seq[:, 0] = X_prime
        # task_space_ctrl_seq[:, 1] = Y_prime
        # ---
        for i in range(len(mass_list)):
            print("i={}: mass={}".format(i, mass_list[i]))
            rectangular_object_xml_generation(config_xml, mass=mass_list[i])
            # ---
            env.set_xml_path(xml_path)
            env.load_model()
            env.reset(init_state)
            env.render()
            # ---
            # env.view()
            img = []
            # import ipdb; ipdb.set_trace()
            for t in range(num_task_step):
                # print("step t = {}".format(t))
                render = env.render()    # ; print(render)
                state  = env.get_state() # ; print(state)
                env.view()
                # ---
                env.set_task_space_ctrl(TaskSpaceValueObject(NTD(task_space_ctrl_seq[t])))
                env.step(is_view=True)
                # import ipdb; ipdb.set_trace()
                # if (t == 6):
                #     import ipdb; ipdb.set_trace()
            render = env.render()
            env.view()
            # import ipdb; ipdb.set_trace()

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
        [0.1,  0.01, 0.1,  0.01]  , # 4
        [0.1,  0.2, 0.2,  0.01]  , # 4
    ]

    # mass_list = [
        # [0.1,  0.01, 0.01, 0.01]  , # 1
        # [0.1,  0.1,  0.01, 0.01]  , # 2
        # [0.1,  0.1,  0.1,  0.01]  , # 3
        # [0.1,  0.01, 0.1,  0.01]  , # 4
        # [0.1,  0.1,  0.01, 0.1]   , # 5
        # # # # ---
        # [0.1, 0.1, 0.1, 0.1]     , # 6
        # [0.01, 0.1, 0.1, 0.01]   , # 7
        # [0.1, 0.01, 0.01, 0.1]   , # 8
        # [0.01, 0.01, 0.01, 0.01] , # 9
        # # ---
        # [0.01, 0.1,  0.1,  0.1]  , # 10
        # [0.01, 0.01, 0.1,  0.1]  , # 11
        # [0.01, 0.01, 0.01, 0.1]  , # 12
        # [0.01, 0.1, 0.01, 0.1]   , # 13
        # [0.1, 0.01, 0.1, 0.1]    , # 14
    # ]
    # ---
    builder       = DomainObjectBuilder()
    director      = CheckEnvRolloutDirector()
    domain_object = director.construct(builder, env_name=env_name)
    CheckEnvDataset.run(domain_object, mass_list)
