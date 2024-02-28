import os
import torch
import pathlib
from copy import deepcopy
from pprint import pprint
from typing import TypedDict
from omegaconf import OmegaConf, DictConfig

# print('hoge')
# piyo = DictConfig
# print('piyo')
# import ipdb; ipdb.set_trace()

class DomainObjectBuilder:
    def __init__(self, config_eval: DictConfig = None):
        # import ipdb; ipdb.set_trace()
        # print('fuga')
        from .DomainObject import DomainObject
        self.domain_object = DomainObject()
        # ---
        # self.domain_object.set_config_eval(config_eval)

    '''
        config
    '''
    def build_config_loader(self):
        from config_loader import ConfigLoader
        self.domain_object.set_config_loader(ConfigLoader())

    def build_config_model(self, config_model = None):
        if config_model is None:
            config_model = self.domain_object.configLoader.load_model()
        # ----
        self.domain_object.set_config_model(config_model)
        self.domain_object.set_config_datamodule(self.domain_object.config_model.datamodule)

    def build_config_ensemble(self):
        self.domain_object.set_config_ensemble(self.domain_object.configLoader.load_ensemble())
        self.domain_object.set_config_datamodule(self.domain_object.config_ensemble.datamodule)

    def build_config_env(self, env_name: str):
        self.domain_object.set_config_env(self.domain_object.configLoader.load_env(env_name))

    def build_config_icem_single(self, env_name: str):
        self.domain_object.set_config_icem(self.domain_object.configLoader.load_icem_single(env_name))

    def build_config_icem_sub(self, env_name: str):
        self.domain_object.set_config_icem(self.domain_object.configLoader.load_icem_sub(env_name))

    def build_config_reference(self, env_name: str):
        self.domain_object.set_config_reference(self.domain_object.configLoader.load_reference(env_name))

    def build_config_xml_generation(self, env_name: str):
        self.domain_object.set_config_xml_generation(self.domain_object.configLoader.load_xml_generation(env_name))

    '''
        config
    '''
    def build_test_config_datamodule(self, config_datamodule: DictConfig = None):
        # if config_datamodule is None:
        #     self.domain_object.set_config_datamodule(self.domain_object.configLoader.load_test_datamodule())
        # else:
        self.domain_object.set_config_datamodule(config_datamodule)

    '''
        model
    '''
    def build_task_space(self, env_name: str, mode: str):
        from task_space import TaskSpaceBuilder
        task_space = TaskSpaceBuilder().build(env_name, mode)
        self.domain_object.set_TaskSpaceValueObject(task_space["TaskSpacePosition"])
        self.domain_object.set_TaskSpaceDiffValueObject(task_space["TaskSpaceDiffPosition"])

    def build_adapter(self, env_name: str):
        from domain_adapter import AdapterFactory, AdapterParamsDict
        paramsDict = AdapterParamsDict(
            env_name                 = env_name,
            config_env               = self.domain_object.config_env,
            config_datamodule        = self.domain_object.config_datamodule,
            to_tensor                = False,
            TaskSpaceValueObject     = self.domain_object.TaskSpaceValueObject,
            TaskSpaceDiffValueObject = self.domain_object.TaskSpaceDiffValueObject,
        )
        adapter = AdapterFactory.create(env_name, paramsDict)
        adapter.set_joint_model_adapter(minmax_info=self.domain_object.trajectory_evaluator.get_minmax_info())
        self.domain_object.set_adapter(adapter)

    def build_image_logger(self):
        from image_logger import ImageLogger
        image_logger = ImageLogger(max_save_num=12, rgb=True)
        self.domain_object.set_image_logger(image_logger)

    def build_ensemble_adapter(self, model_dir):
        from domain_adapter import EnsembleAdapter
        ensemble_adapter = EnsembleAdapter()
        load_path        = os.path.join(model_dir, "ensemble_dataset", "statistics", "statistics.pth")
        minmax_info      = torch.load(load_path)
        ensemble_adapter.set_joint_model_adapter(minmax_info=minmax_info)
        self.domain_object.set_ensemble_adapter(ensemble_adapter)

    def build_model_domain_object(self, config: DictConfig = None):
        from domain_object_builder.model import DomainObjectBuilderFactor
        from domain_object_director.model import DomainObjectDirectorFactory
        # ---
        if config is None:
            config = self.domain_object.config_model
        # ---
        model_class         = self.domain_object.config_model.model_class
        model_builder       = DomainObjectBuilderFactor.create(model_class, config)
        model_director      = DomainObjectDirectorFactory.create(model_class)
        model_domain_object = model_director.construct(builder=model_builder)
        # ---
        self.domain_object.set_model_domain_object(model_domain_object)


    def build_model_dir(self, model_dir: str):
        self.domain_object.set_model_dir(model_dir)

    def build_model(self):
        from cdsvae.domain.model import ModelFactory
        model = ModelFactory.create(
            name          = self.domain_object.config_model.model_class,
            domain_object = self.domain_object.model_domain_object,
        )
        self.domain_object.set_model(model)

    def build_lit_model_train(self):
        from domain_object_builder.lit_model_load import LitModelLoadDomainObjectBuilder
        from domain_object_director.lit_model_load import LitModelTrainingLoadDomainObjectDirector
        # ---
        builder   = LitModelLoadDomainObjectBuilder()
        director  = LitModelTrainingLoadDomainObjectDirector()
        config    = self.domain_object.config_model
        lit_model = director.construct(builder, self.domain_object.model, config)
        self.domain_object.set_lit_model(lit_model)

    def build_lit_model_eval(self, config: DictConfig):
        from domain_object_builder.lit_model_load import LitModelLoadDomainObjectBuilder
        from domain_object_director.lit_model_load import LitModelEvalLoadDomainObjectDirector
        # ---
        builder   = LitModelLoadDomainObjectBuilder()
        director  = LitModelEvalLoadDomainObjectDirector()
        lit_model = director.construct(builder, self.domain_object.model, config)
        self.domain_object.set_lit_model(lit_model)

    def build_filtering_model(self):
        from cdsvae.domain.filtering import Filtering
        filtering_model = Filtering(self.domain_object.model_domain_object)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        filtering_model.to(device)
        filtering_model.eval()
        self.domain_object.set_filtering_model(filtering_model)

    def build_prediction_model(self):
        from cdsvae.domain.prediction import Prediction
        prediction_model = Prediction(self.domain_object.model_domain_object)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prediction_model.to(device)
        prediction_model.eval()
        self.domain_object.set_prediction_model(prediction_model)

    '''
        データモジュール
        ・cdsvaeとensembleモデルでどっちも同じ手続きを使いたいのでselfのdomain_object経由ではなく
        引数でどっちを入れるか選択できるようにする
    '''
    def build_model_datamodule(self, config_datamodule: DictConfig = None):
        from cdsvae.domain.datamodule import DataModuleFactory, DatamoduleParamsDict
        # ---
        if config_datamodule is None:
            config_datamodule = self.domain_object.config_model.datamodule
        # ---
        paramsDict = DatamoduleParamsDict(config_datamodule=config_datamodule, adapter=self.domain_object.adapter)
        datamodule = DataModuleFactory.create(name=config_datamodule.name, paramsDict=paramsDict)
        self.domain_object.set_datamodule(datamodule)

    def build_ensemble_datamodule(self, config: DictConfig, model_dir: str):
        from cdsvae.domain.datamodule import DatamoduleParamsDict
        from cdsvae.domain.datamodule.ensemble import EnsembleDataModule
        # ---
        assert config.datamodule.name == "ensemble"
        config.datamodule.dataset_dir  = model_dir
        config.datamodule.dataset_name = "ensemble_dataset"
        # ---
        paramsDict = DatamoduleParamsDict(config_datamodule=config.datamodule, adapter=self.domain_object.ensemble_adapter)
        datamodule = EnsembleDataModule(paramsDict=paramsDict)
        self.domain_object.set_datamodule(datamodule)

    '''
        トレーニングモジュール
        ・cdsvaeとensembleモデルでどっちも同じ手続きを使いたいのでselfのdomain_object経由ではなく
        引数でどっちを入れるか選択できるようにする
    '''
    def build_training_modules(self, config_model: DictConfig, version : str = None):
        from .model_training import ModelTrainingDomainObjectBuilder
        builder = ModelTrainingDomainObjectBuilder(config_model=config_model, datamodule=self.domain_object.datamodule)
        builder.build_tb_logger(version=version)
        builder.build_trainer()
        # ---
        model_training_domain_object = builder.get_domain_object()
        # ---
        self.domain_object.set_model_dir(model_training_domain_object.trainer.log_dir)
        # ---
        self.domain_object.set_datamodule(model_training_domain_object.datamodule)
        self.domain_object.set_tb_logger(model_training_domain_object.tb_logger)
        self.domain_object.set_trainer(model_training_domain_object.trainer)


    '''
        build_ensemble
    '''
    def build_ensemble(self, config: DictConfig=None):
        from cdsvae.domain.model.ensemble import EnsembleFullConnect
        # ---
        if config is None:
            config = self.domain_object.config_ensemble
        # ---
        model = EnsembleFullConnect(**config.model, num_ensemble=config.num_ensemble)
        self.domain_object.set_ensemble(model)

    def build_lit_ensemble(self, model_dir: str):
        from domain_object_builder.lit_ensemble_load import LitEnsembleLoadDomainObjectBuilder
        from domain_object_director.lit_ensemble_load import LitEnsembleTrainingLoadDomainObjectDirector
        # ---
        self.domain_object.config_ensemble.logger.save_dir = model_dir
        # ---
        builder   = LitEnsembleLoadDomainObjectBuilder()
        director  = LitEnsembleTrainingLoadDomainObjectDirector()
        lit_model = director.construct(builder, self.domain_object.ensemble, config_model=self.domain_object.config_ensemble)
        self.domain_object.set_lit_ensemble(lit_model)

    def build_lit_ensemble_eval(self, config: DictConfig):
        from domain_object_builder.lit_ensemble_load import LitEnsembleLoadDomainObjectBuilder
        from domain_object_director.lit_ensemble_load import LitEnsembleEvalLoadDomainObjectDirector
        # ---
        builder   = LitEnsembleLoadDomainObjectBuilder()
        director  = LitEnsembleEvalLoadDomainObjectDirector()
        lit_model = director.construct(builder, self.domain_object.ensemble, config_model=config)
        self.domain_object.set_lit_ensemble(lit_model)


    def build_nominal_ctrl(self, config_model):
        from robel_dclaw_env.domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
        repository = Repository(**self.domain_object.config_cdsvae_test.nominal, read_only= True)
        repository.open(filename="nominal")
        nominal_ctrl_joint_space_position = repository.repository["ctrl"][config_model.datamodule.data_type.ctrl_type]
        u_nominal = self.domain_object.adapter.envCtrl2modelCtrl(nominal_ctrl_joint_space_position)
        repository.close()
        self.domain_object.set_nominal_ctrl(u_nominal)

    def build_control_adaptor(self):
        from icem_torch.control_adaptor import ControlAdaptorFactory
        control_adaptor = ControlAdaptorFactory.create(
            env_name      = self.domain_object.config_env.env.env_name,
            TaskSpaceDiff = self.domain_object.TaskSpaceDiffValueObject,
        )
        self.domain_object.set_control_adaptor(control_adaptor)

    def build_env_planning(self):
        from planning.env_planning import EnvPlanningUnderTaskSpaceDiff
        from .PlanningDomainObjectDict import PlanningDomainObjectDict
        planning_domain_object_dict = PlanningDomainObjectDict(
            env_object      = self.domain_object.env_object,
            config_env      = self.domain_object.config_env,
            TaskSpaceDiff   = self.domain_object.TaskSpaceDiffValueObject,
            save_dir        = self.domain_object.env_data_repository.save_dir,
            control_adaptor = self.domain_object.control_adaptor,

        )
        self.domain_object.set_planning_domain_object_dict(planning_domain_object_dict)
        self.domain_object.set_planning(EnvPlanningUnderTaskSpaceDiff)

    def build_model_planning(self):
        from planning.model_planning import ModelPlanningParamesDict
        # from planning.model_planning import ModelPlanningWithEnsembleUnderTaskSpace
        from planning.model_planning import ModelPlanningUnderTaskSpace
        paramsDict = ModelPlanningParamesDict(
            model                    = self.domain_object.lit_model.model,
            # ensemble               = self.domain_object.ensemble,
            adapter                  = self.domain_object.adapter,
            # ensemble_adapter       = self.domain_object.ensemble_adapter,
            planning_visualizer      = self.domain_object.planning_visualizer,
            TaskSpaceDiffValueObject = self.domain_object.TaskSpaceDiffValueObject,
        )
        # model_planning_instance = ModelPlanningWithEnsembleUnderTaskSpace(paramsDict)
        model_planning_instance = ModelPlanningUnderTaskSpace(paramsDict)
        self.domain_object.set_planning(model_planning_instance)


    def build_model_planning_with_abs_ctrl(self):
        from planning.model_planning import ModelPlanningParamesDict
        from planning.model_planning import ModelPlanningUnderAbsoluteValueTaskSpace
        paramsDict = ModelPlanningParamesDict(
            model                    = self.domain_object.lit_model.model,
            adapter                  = self.domain_object.adapter,
            TaskSpaceDiffValueObject = self.domain_object.TaskSpaceDiffValueObject,
        )
        model_planning_instance = ModelPlanningUnderAbsoluteValueTaskSpace(paramsDict)
        self.domain_object.set_planning(model_planning_instance)


    def build_model_planning_with_abs_ctrl_fixed_init_ctrl(self):
        from planning.model_planning import ModelPlanningParamesDict
        from planning.model_planning import ModelPlanningUnderAbsoluteValueTaskSpace_fixedinitctrl
        paramsDict = ModelPlanningParamesDict(
            model                    = self.domain_object.lit_model.model,
            adapter                  = self.domain_object.adapter,
            TaskSpaceDiffValueObject = self.domain_object.TaskSpaceDiffValueObject,
        )
        model_planning_instance = ModelPlanningUnderAbsoluteValueTaskSpace_fixedinitctrl(paramsDict)
        self.domain_object.set_planning(model_planning_instance)


    def build_model_prediction_planning(self):
        from planning.model_prediction_planning import ModelPredictionPlanningInstanceParamesDict
        from planning.model_prediction_planning import ModelPredictionPlanningDebug
        paramsDict = ModelPredictionPlanningInstanceParamesDict(
            model                    = self.domain_object.lit_model.model,
            adapter                  = self.domain_object.adapter,
            TaskSpaceDiffValueObject = self.domain_object.TaskSpaceDiffValueObject,
        )
        model_prediction_planning_instance = ModelPredictionPlanningDebug(paramsDict)
        self.domain_object.set_planning(model_prediction_planning_instance)

    def build_planning_visualizer(self):
        from planning.visualizer import PlanningVisualizer
        planning_visualizer = PlanningVisualizer(
            env_name  = self.domain_object.config_env.env.env_name,
            adapter   = self.domain_object.adapter,
            reference = self.domain_object.reference,
            save_dir  = self.domain_object.icem_repository.save_dir,
        )
        self.domain_object.set_planning_visualizer(planning_visualizer)

    def build_trajectory_evaluator(self, env_name: str = None):
        from trajectory_evaluator import TrajectoryEvaluatorFactory
        env_name             = self.domain_object.config_env.env.env_name if env_name is None else env_name
        trajectory_evaluator = TrajectoryEvaluatorFactory.create(env_name)
        self.domain_object.set_trajectory_evaluator(trajectory_evaluator)

    def build_data_collection_planning(self):
        from planning.data_collection_planning import DataCollectionPlanningDomainObjectDict
        # from planning.data_collection_planning import DataCollectionPlanning_for_loop
        from planning.data_collection_planning import DataCollectionPlanning_while_loop
        # ---
        planning_domain_object_dict = DataCollectionPlanningDomainObjectDict(
            env_object             = self.domain_object.env_object,
            config_env             = self.domain_object.config_env,
            xml_path               = self.domain_object.original_xml_path,
            TaskSpaceDiff          = self.domain_object.TaskSpaceDiffValueObject,
            control_adaptor        = self.domain_object.control_adaptor,
            env_data_container     = self.domain_object.env_data_container,
            env_data_repository    = self.domain_object.env_data_repository,
            trajectory_evaluator   = self.domain_object.trajectory_evaluator,
            population_sampler     = self.domain_object.population_sampler,
        )
        self.domain_object.set_planning_domain_object_dict(planning_domain_object_dict)
        self.domain_object.set_planning(DataCollectionPlanning_while_loop)

    def build_data_collection_planning_with_fixed_init_ctrl(self):
        from planning.data_collection_planning import DataCollectionPlanningDomainObjectDict
        from planning.data_collection_planning import DataCollectionPlanning_while_loop_with_fixed_init_ctrl
        # ---
        planning_domain_object_dict = DataCollectionPlanningDomainObjectDict(
            env_object             = self.domain_object.env_object,
            config_env             = self.domain_object.config_env,
            config_icem            = self.domain_object.config_icem,
            xml_path               = self.domain_object.original_xml_path,
            TaskSpaceDiff          = self.domain_object.TaskSpaceDiffValueObject,
            control_adaptor        = self.domain_object.control_adaptor,
            env_data_container     = self.domain_object.env_data_container,
            env_data_repository    = self.domain_object.env_data_repository,
            trajectory_evaluator   = self.domain_object.trajectory_evaluator,
            population_sampler     = self.domain_object.population_sampler,
        )
        self.domain_object.set_planning_domain_object_dict(planning_domain_object_dict)
        self.domain_object.set_planning(DataCollectionPlanning_while_loop_with_fixed_init_ctrl)


    def build_loaded_data_collection_planning(self):
        from planning.data_collection_planning import DataCollectionPlanningDomainObjectDict
        from planning.data_collection_planning import DataCollectionPlanning_for_loaded_data_collection
        # ---
        planning_domain_object_dict = DataCollectionPlanningDomainObjectDict(
            env_object             = self.domain_object.env_object,
            config_env             = self.domain_object.config_env,
            xml_path               = self.domain_object.original_xml_path,
            # TaskSpaceDiff          = self.domain_object.TaskSpaceDiffValueObject,
            TaskSpaceAbs           = self.domain_object.TaskSpaceValueObject,
            control_adaptor        = self.domain_object.control_adaptor,
            env_data_container     = self.domain_object.env_data_container,
            env_data_repository    = self.domain_object.env_data_repository,
            trajectory_evaluator   = self.domain_object.trajectory_evaluator,
            population_sampler     = self.domain_object.population_sampler,
        )
        self.domain_object.set_planning_domain_object_dict(planning_domain_object_dict)
        self.domain_object.set_planning(DataCollectionPlanning_for_loaded_data_collection)



    def build_reference(self, paramsDict: TypedDict = None):
        from reference import ReferenceFactory
        from reference import ReferenceParamsDictFactory
        # ---
        env_name = self.domain_object.config_env.env.env_name
        if paramsDict is None:
            assert False
            ParamsDictObject = ReferenceParamsDictFactory.create(env_name)
            paramsDict       = ParamsDictObject(**self.domain_object.config_reference.env)
        # ---
        reference = ReferenceFactory.create(
            env_name             = env_name,
            planning_horizon     = self.domain_object.config_icem.icem.planning_horizon,
            task_horizon         = self.domain_object.config_reference.task_horizon,
            object_position_init = self.domain_object.config_env.env.init_state.object_position,
            paramsDict           = paramsDict,
            adapter              = self.domain_object.adapter,
        )
        assert "pushing" in self.domain_object.config_env.env.env_name # バルブの方の実装TrajectoryEvaluatorが未実装でmin, maxがないので
        minmax_info = self.domain_object.trajectory_evaluator.get_minmax_info()
        reference.set_normalize_parameter(
            x_min = minmax_info["min_vertical"],
            x_max = minmax_info["max_vertical"],
            m     = 0.0,
            M     = 1.0,
        )
        self.domain_object.set_reference(reference)

    def build_cost_model(self):
        from cost.model_planning_cost import CostModelFactory
        cost = CostModelFactory.create(self.domain_object.config_env.env.env_name, self.domain_object.config_icem.cost)
        self.domain_object.set_cost_model(cost)

    def build_cost_env(self):
        from cost.env_planning_cost import CostEnvFactory
        cost = CostEnvFactory.create(self.domain_object.config_env.env.env_name, self.domain_object.config_icem.cost)
        self.domain_object.set_cost_env(cost)

    def build_cost_data_collection(self):
        from cost.env_planning_cost import CostEnvFactory
        cost = CostEnvFactory.create(env_name="data_collection", config=self.domain_object.config_icem.cost)
        self.domain_object.set_cost_env(cost)

    def build_icem_repository(self, model_dir:str):
        '''
        従来は build_icem の中に一緒に書いていたが, planningで予測軌道を保存する際に
        planning内部に保存先のディレクトリを渡そうとするとicemのインスタンス化の前に
        icem_repositoryのsave_dirをplanningオブジェクトに渡す必要が生じるたので分離
        '''
        from icem_torch import iCEM_Repository
        icem_repository = iCEM_Repository(model_dir, tag=self.domain_object.config_eval.tag)
        icem_repository.set_config_and_repository(self.domain_object.config_icem)
        icem_repository.save_config()
        self.domain_object.set_icem_repository(icem_repository)

    def build_icem(self):
        from icem_torch import iCEMBuilder
        icem = iCEMBuilder.build(
            planning   = self.domain_object.planning,
            cost       = self.domain_object.cost_model,
            config     = self.domain_object.config_icem.icem,
            repository = self.domain_object.icem_repository,
        )
        self.domain_object.set_icem(icem)

    def build_population_sampler(self):
        from icem_torch.icem_subparticle.population import ColoredPopulationSampler
        from icem_torch.icem_subparticle.population import ColoredPopulationSamplerParamsDict
        from icem_torch.icem_subparticle.population import PopulationSampingDistribution
        from icem_torch.icem_subparticle.population import ColoredNoiseSampler
        paramsDict = ColoredPopulationSamplerParamsDict(
            sampling_dist          = PopulationSampingDistribution(self.domain_object.config_icem.icem),
            colored_noise_exponent = self.domain_object.config_icem.icem.colored_noise_exponent,
            colored_noise_sampler  = ColoredNoiseSampler(self.domain_object.config_icem.icem),
            lower_bound_sampling   = self.domain_object.config_icem.icem.lower_bound_sampling,
            upper_bound_sampling   = self.domain_object.config_icem.icem.upper_bound_sampling,
        )
        population_sampler = ColoredPopulationSampler(paramsDict)
        self.domain_object.set_population_sampler(population_sampler)

    def build_icem_multiprocessing(self):
        from icem_torch.doi_extractor import ObjectDimensionOfInterestFactory
        from icem_torch.visualization import Visualizer
        from icem_torch.icem_subparticle import CostManager
        from icem_torch.icem_subparticle import iCEM_Subparticle_ParamDict
        from icem_torch.icem_subparticle import iCEM_Subparticle_Factory
        # ---
        cost_manager  = CostManager(cost=self.domain_object.cost_env, verbose=True)
        doi_extractor = ObjectDimensionOfInterestFactory.create(self.domain_object.config_env.env.env_name)
        visualizer    = Visualizer(
            env_name      = self.domain_object.config_env.env.env_name,
            reference     = self.domain_object.reference,
            save_dir      = self.domain_object.env_data_repository.save_dir,
            doi_extractor = doi_extractor,
        )
        self.domain_object.cost_env.set_num_divide(num_divide=self.domain_object.config_icem.icem.num_sample)
        paramsDict = iCEM_Subparticle_ParamDict(
            planning           = self.domain_object.planning,
            domain_object_dict = self.domain_object.planning_domain_object_dict,
            cost_manager       = cost_manager,
            config             = self.domain_object.config_icem.icem,
            visualizer         = visualizer,
            population_sampler = self.domain_object.population_sampler,
        )
        icem = iCEM_Subparticle_Factory.create(paramsDict, mockup=self.domain_object.config_icem.mockup)
        self.domain_object.set_icem(icem)

    def build_icem_multiprocessing_random_data_collection(self):
        from icem_torch.icem_subparticle import iCEM_Subparticle_ParamDict
        from icem_torch.icem_subparticle import iCEM_Subparticle_RandomDataCollection
        paramsDict = iCEM_Subparticle_ParamDict(
            planning           = self.domain_object.planning,
            domain_object_dict = self.domain_object.planning_domain_object_dict,
            cost_manager       = None,
            config             = self.domain_object.config_icem.icem,
            visualizer         = None,
            population_sampler = self.domain_object.population_sampler,
        )
        icem = iCEM_Subparticle_RandomDataCollection(paramsDict)
        self.domain_object.set_icem(icem)


    def build_icem_multiprocessing_random_data_collection_with_fixed_initial_motion(self):
        from icem_torch.icem_subparticle import iCEM_Subparticle_ParamDict
        from icem_torch.icem_subparticle import iCEM_Subparticle_RandomDataCollection_with_fixed_initial_motion
        paramsDict = iCEM_Subparticle_ParamDict(
            planning           = self.domain_object.planning,
            domain_object_dict = self.domain_object.planning_domain_object_dict,
            cost_manager       = None,
            config             = self.domain_object.config_icem.icem,
            visualizer         = None,
            population_sampler = self.domain_object.population_sampler,
        )
        icem = iCEM_Subparticle_RandomDataCollection_with_fixed_initial_motion(paramsDict)
        self.domain_object.set_icem(icem)


    def build_icem_multiprocessing_loaded_data_collection(self):
        from icem_torch.icem_subparticle import iCEM_Subparticle_ParamDict
        from icem_torch.icem_subparticle import iCEM_Subparticle_LoadedDataCollection
        paramsDict = iCEM_Subparticle_ParamDict(
            planning           = self.domain_object.planning,
            domain_object_dict = self.domain_object.planning_domain_object_dict,
            cost_manager       = None,
            config             = self.domain_object.config_icem.icem,
            visualizer         = None,
            population_sampler = self.domain_object.population_sampler,
        )
        icem = iCEM_Subparticle_LoadedDataCollection(paramsDict)
        self.domain_object.set_icem(icem)


    def build_model_file_manager(self):
        from robel_dclaw_env.domain.model_path_manager import ModelPathManager
        model_file_manager = ModelPathManager()
        self.domain_object.set_model_file_manager(model_file_manager)

    def build_original_xml_path(self):
        original_xml_path = os.path.join(
            self.domain_object.config_env.env.model_dir,
            self.domain_object.config_env.env.model_file,
        )
        self.domain_object.set_original_xml_path(original_xml_path)

    def build_xml_model_modifier(self):
        from robel_dclaw_env.domain import XMLModelFileModifier
        xml_modifier = XMLModelFileModifier()
        self.domain_object.set_xml_modifier(xml_modifier)

    def build_env_adapter(self):
        from robel_dclaw_env.domain.environment import EnvironmentBuilder
        from environments import ModelAdaptedEnvironment
        env_dict    = EnvironmentBuilder().build(self.domain_object.config_env)
        init_state  = env_dict["init_state"]
        env_adapter = ModelAdaptedEnvironment(env_dict["env"], self.domain_object.adapter)
        self.domain_object.set_env_adapter(env_adapter)
        self.domain_object.set_env_init_state(init_state)

    def build_env_instance(self):
        from robel_dclaw_env.domain.environment import EnvironmentBuilder
        env_dict = EnvironmentBuilder().build(self.domain_object.config_env)
        self.domain_object.set_env(env_dict["env"])
        self.domain_object.set_env_init_state(env_dict["init_state"])
        self.domain_object.set_TaskSpaceValueObject(env_dict["TaskSpacePosition"])
        self.domain_object.set_TaskSpaceDiffValueObject(env_dict["TaskSpaceDiffPosition"])

    def build_env_object(self):
        from robel_dclaw_env.domain.environment import EnvironmentFactory
        from task_space import TaskSpaceBuilder
        env_object, state_factory = EnvironmentFactory.create(self.domain_object.config_env.env.env_name)
        task_space = TaskSpaceBuilder().build(self.domain_object.config_env.env.env_name, mode="torch")
        init_state = state_factory.create_for_init_env(task_space["TaskSpacePosition"], **self.domain_object.config_env.env.init_state)
        self.domain_object.set_env_object(env_object)
        self.domain_object.set_env_init_state(init_state)
        self.domain_object.set_TaskSpaceValueObject(task_space["TaskSpacePosition"])
        self.domain_object.set_TaskSpaceDiffValueObject(task_space["TaskSpaceDiffPosition"])

    def build_file_copy_manager(self, dataset_name: str = None):
        from file_copy_manager import FileCopyManager
        # ---
        if dataset_name is not None:
            self.domain_object.config_icem.save.dataset_name = dataset_name
        # ---
        file_copy_manager = FileCopyManager(self.domain_object.config_icem)
        self.domain_object.set_file_copy_manager(file_copy_manager)

    def build_xml_generator(self, save_dir: str = None):
        env_name = self.domain_object.config_env.env.env_name
        if "pushing" not in env_name: return # pushing環境でないならxml生成はしない
        from xml_generation import XMLModelGeneratorFactory, ObjectParamsFactory
        if save_dir is None:
            save_dir = '/nfs/monorepo_ral2023/robel_dclaw_env/robel_dclaw_env/domain/environment/model/pushing_object'
        ObjectParamDict = ObjectParamsFactory.create(object_type=self.domain_object.config_xml_generation.object_type.name)
        object_params   = ObjectParamDict(**self.domain_object.config_xml_generation.object_type)
        xml_generator   = XMLModelGeneratorFactory.create(
            object_type   = self.domain_object.config_xml_generation.object_type.name,
            object_params = object_params,
            save_dir      = save_dir,
        )
        self.domain_object.set_xml_generator(xml_generator)

    def build_usecase_repository(self):
        from usecase_repository import UsecaseRepository
        usecaseRepository = UsecaseRepository()
        usecaseRepository.set_image_save_dir(self.domain_object.icem.repository.save_dir)
        self.domain_object.set_usecase_repository(usecaseRepository)

    def build_filtering_manager(self):
        from filtering_manager import FilteringManager
        filtering_manager = FilteringManager(self.domain_object.filtering_model, num_batch=1)
        self.domain_object.set_filtering_manager(filtering_manager)

    def build_prediction_manager(self):
        from prediction_manager import PredictionManager
        prediction_manager = PredictionManager(self.domain_object.prediction_model, num_batch=1)
        self.domain_object.set_prediction_manager(prediction_manager)

    def build_icem_manager(self):
        from icem_manager import iCEMManager
        icem_manager = iCEMManager(self.domain_object.icem, self.domain_object.reference)
        self.domain_object.set_icem_manager(icem_manager)

    def build_icem_subparticle_manager(self):
        from icem_manager import iCEM_SubparticleManager
        icem_subparticle_manager = iCEM_SubparticleManager(self.domain_object.icem, self.domain_object.reference)
        self.domain_object.set_icem_subparticle_manager(icem_subparticle_manager)

    def build_mpc_data_logger(self):
        from data_logger import DataLogger
        from data_logger.data_container import DataContainer
        from data_logger.cost_calculator import CostCalculatorFactory
        # ---
        env_name  = self.domain_object.config_env.env.env_name
        adapter   = self.domain_object.adapter
        reference = self.domain_object.reference
        save_dir  = self.domain_object.icem.repository.save_dir
        # ---
        data_logger = DataLogger(
            data_container  = DataContainer(adapter=adapter, reference=reference),
            cost_calculator = CostCalculatorFactory.create(env_name, reference, adapter, save_dir),
            visualizer      = self.domain_object.mpc_result_visualizer,
            save_dir        = save_dir,
        )
        self.domain_object.set_data_logger(data_logger)

    def build_env_data_container(self):
        from data_logger.env_data_logger import EnvDataContainer
        self.domain_object.set_env_data_container(EnvDataContainer())

    def build_ensemble_data_container(self):
        from data_logger.ensemble_data_logger import EnsembleDataContainer
        self.domain_object.set_ensemble_data_container(EnsembleDataContainer())

    def build_env_data_repository(self, dataset_dir:str = None, read_only: bool = False):
        from data_logger.env_data_logger import EnvDataLoggerParams, create_env_dataset_dir_name
        from data_logger.env_data_logger import EnvDataRepository
        if dataset_dir is None:
            dataset_dir = create_env_dataset_dir_name(
                config_save = self.domain_object.config_icem.save,
                config_icem = self.domain_object.config_icem.icem,
            )
        params     = EnvDataLoggerParams(dataset_dir=dataset_dir, read_only=read_only)
        repository = EnvDataRepository(params)
        self.domain_object.set_env_data_repository(repository)

    def build_shelve_repository(self, save_dir: str = None, read_only: bool = False):
        from data_logger.env_data_logger import ShelveRepository
        shelve_repository = ShelveRepository(save_dir, read_only)
        self.domain_object.set_shelve_repository(shelve_repository)

    def build_env_data_stack(self):
        from data_logger.env_data_logger import EnvDataStack
        self.domain_object.set_env_data_stack(EnvDataStack())

    def build_ensemble_data_stack(self):
        from data_logger.ensemble_data_logger import EnsembleDataStack
        self.domain_object.set_ensemble_data_stack(EnsembleDataStack())

    def build_env_data_stack_info_writer(self):
        from data_logger.env_data_logger import EnvDataStackInfoWriter
        self.domain_object.set_env_data_stack_info_writer(EnvDataStackInfoWriter())

    def build_ensemble_data_stack_info_writer(self):
        from data_logger.ensemble_data_logger import EnsembleDataStackInfoWriter
        self.domain_object.set_ensemble_data_stack_info_writer(EnsembleDataStackInfoWriter())

    def build_replay(self):
        from replay import Replay
        replay = Replay(self.domain_object.planning, self.domain_object.model, self.domain_object.adapter, duration=250)
        self.domain_object.set_replay(replay)

    def build_dataset_visualizer(self, save_dir: str):
        from dataset_visualizer import DatasetVisualizer
        dataset_visualizer = DatasetVisualizer(
            env_name = self.domain_object.config_env.env.env_name,
            save_dir = save_dir,
        )
        self.domain_object.set_dataset_visualizer(dataset_visualizer)

    def build_ensemble_dataset_visualizer(self, save_dir: str):
        from dataset_visualizer import EnsembleDatasetVisualizer
        self.domain_object.set_ensemble_dataset_visualizer(EnsembleDatasetVisualizer(save_dir))

    def build_image_viewer(self):
        from image_viewer import ImageViewer
        image_viewer = ImageViewer(window_size=(500, 500), rgb=True)
        self.domain_object.set_image_viewer(image_viewer)

    def build_mpc_result_visualizer(self):
        from mpc_result_visualizer import MPCResultVisualizerBuilder, MPCResultVisualizerDirectorFactory, MPCResultVisualizerFactory
        builder       = MPCResultVisualizerBuilder(save_dir=self.domain_object.icem_repository.save_dir)
        director      = MPCResultVisualizerDirectorFactory.create(
            env_name              = self.domain_object.config_env.env.env_name,
            builder               = builder,
        )
        domain_object = director.construct()
        mpc_result_visualizer = MPCResultVisualizerFactory.create(
            env_name      = self.domain_object.config_env.env.env_name,
            reference     = self.domain_object.reference,
            adapter       = self.domain_object.adapter,
            domain_object = domain_object,
        )
        self.domain_object.set_mpc_result_visualizer(mpc_result_visualizer)

    def build_tsne_visualizer(self):
        from visualization.tsne_visualizer import TSNEVisualizer
        tsne_visualizer = TSNEVisualizer()
        self.domain_object.set_tsne_visualizer(tsne_visualizer)

    def build_scatter_visualizer(self, figsize=(6,6)):
        from visualization.scatter import ScatterVisualizer
        visualizer = ScatterVisualizer(figsize=figsize)
        self.domain_object.set_scatter_visualizer(visualizer)

    def get_domain_object(self):
        '''
            - return deepcopy(self.domain_object) はダメ
            - 並列化する場合などQueueを使っている場合deepcopyで直列化できない
        '''
        return self.domain_object
