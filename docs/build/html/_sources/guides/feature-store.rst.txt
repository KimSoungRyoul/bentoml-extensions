===========
Feature Store
===========

...


About FeatureStore Repository
------------------------------------

W todo ...


FeatureStore Runner (BentoML <=1.1.x)
------------

...

.. tab-set::

    .. tab-item:: Redis

        ...

        .. code-block:: python

            import logging
            from typing import TypedDict

            import bentoml
            from bentoml.io import NumpyNdarray, JSON

            import bentomlx

            logger = logging.getLogger("bentoml")

            reg_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()

            redis_settings = {
                "HOST": "localhost",
                "PORT": 6379,
                "DB": 0,
            }


            class RedisRecord(TypedDict):
                pk: str
                feature1: float
                feature2: float


            repo_runner = bentomlx.featurestore.runner.redis(
                db_settings=redis_settings, record_class=RedisRecord, embedded=True
            )

            svc = bentoml.Service("linear_regression", runners=[reg_runner, repo_runner])


            @svc.api(input=JSON.from_sample(["feature_key1", "feature_key2", "feature_key3"]),
                     output=NumpyNdarray.from_sample([[0.111, 0.222, 0.333]]))
            async def predict(feature_key_list: list[str]):
                np_arr = await repo_runner.get_many.async_run(pks=feature_key_list, np=True)  # np=True -> return ndarray
                result = await reg_runner.predict.async_run(np_arr)
                return result


    .. tab-item:: Aerospike

        Aerospike Generally read performance is faster than redis.
        Especially for get data with many keys

        .. code-block:: python

            import logging

            import bentoml
            from bentoml.io import NumpyNdarray, JSON

            import bentomlx

            logger = logging.getLogger("bentoml")

            db_settings = {
                "HOSTS": ["127.0.0.1:3000", ],
                "NAMESPACE": "test",
                "SET_NAME": "sample_feature",
                # "USERNAME": "user",
                # "PASSWORD": " password",
                "USE_SHARED_CONNECTION": False
            }

            reg_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()
            repo_runner = bentomlx.featurestore.runner.aerospike(db_settings=db_settings, embedded=True)

            svc = bentoml.Service("linear_regression", runners=[reg_runner, fs_repo_runner])


            @svc.api(input=JSON.from_sample(["pk1", "pk2", "pk3"]), output=NumpyNdarray.from_sample([[0.111, 0.222, 0.333]]))
            async def predict(feature_key_list: list[str]):
                nd_arr: list[dict[str, float]] = await repo_runner.get_many.async_run(pks=feature_key_list, numpy=True)
                result = await reg_runner.predict.async_run(nd_arr)
                return result


FeatureStore Inner Service (BentoML >=1.2.x)
----------------

...

.. tab-set::

    .. tab-item:: Redis

        ...

        .. code-block:: python

            todo ....


    .. tab-item:: Aerospike

        todo:  there is no way inject db_settings to inner Repository SVC,....

        .. code-block:: python

            import logging

            import bentoml
            import numpy as np
            import numpy.random
            from pydantic import BaseModel, Field

            import bentomlx

            logger = logging.getLogger("bentoml")

            db_settings = {
                "HOSTS": ["127.0.0.1:3000", ],
                "NAMESPACE": "test",
                "SET_NAME": "sample_feature",
                # "USERNAME": "user",
                # "PASSWORD": " password",
                "USE_SHARED_CONNECTION": False
            }

            reg_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()
            LinearRegService = bentoml.runner_service(runner=reg_runner)


            class BentoRequest(BaseModel):
                feature_key_list: list[str] = Field(description="pk list", example=["pk1", "pk2"])


            @bentoml.service(
                workers=1
            )
            class HelloBentoService:
                repo = bentoml.depends(
                    # todo: there is no way inject db_settings to inner Repository SVC....
                    on=bentomlx.featurestore.aerospike.AsyncFeatureRepository
                )
                linear_reg = bentoml.depends(
                    on=LinearRegService
                )

                @bentoml.api(route="/predict")
                async def predict(self, bento_request: BentoRequest) -> np.ndarray:
                    nd_arr: np.ndarray = await self.repo.get_many(pks=bento_request.feature_key_list, _np=True)
                    logger.info(f"Aerospike data: {nd_arr.tolist()}")
                    result = await reg_runner.predict.async_run(nd_arr)
                    return result


Manage models
-------------

Saving a model to the Model Store and retrieving it are the two most common use cases for managing models. In addition to them, you can also perform other operations by using the BentoML CLI or management APIs.

CLI commands
^^^^^^^^^^^^

You can perform the following operations on models by using the BentoML CLI.

.. tab-set::

    .. tab-item:: List

        To list all available models:

        .. code-block:: bash

            $ bentoml models list

            Tag                                   Module  Size      Creation Time
            summarization-model:btwtmvu5kwqc67i3          1.14 GiB  2023-12-18 03:25:10

    .. tab-item:: Get

        To retrieve the information of a specific model:

        .. code-block:: bash

            $ bentoml models get summarization-model:latest

            name: summarization-model
            version: btwtmvu5kwqc67i3
            module: ''
            labels: {}
            options: {}
            metadata:
            model_name: sshleifer/distilbart-cnn-12-6
            task_name: summarization
            context:
            framework_name: ''
            framework_versions: {}
            bentoml_version: 1.1.10.post84+ge2e9ccc1
            python_version: 3.9.16
            signatures: {}
            api_version: v1
            creation_time: '2023-12-18T03:25:10.972481+00:00'

    .. tab-item:: Import/Export

        You can export a model in the BentoML Model Store as a standalone archive file and share it between teams or move it between different build stages. For example:

        .. code-block:: bash

            $ bentoml models export summarization-model:latest .

            Model(tag="summarization-model:btwtmvu5kwqc67i3") exported to ./summarization-model-btwtmvu5kwqc67i3.bentomodel

        .. code-block:: bash

            $ bentoml models import ./summarization-model-btwtmvu5kwqc67i3.bentomodel

            Model(tag="summarization-model:btwtmvu5kwqc67i3") imported

        You can export models to and import models from external storage devices, such as AWS S3, GCS, FTP and Dropbox. For example:

        .. code-block:: bash

            pip install fs-s3fs  *# Additional dependency required for working with s3*
            bentoml models export summarization-model:latest s3://my_bucket/my_prefix/

    .. tab-item:: Pull/Push

        `BentoCloud <https://cloud.bentoml.com/>`_ provides a centralized model repository with flexible APIs and a web console for managing all models created by your team. After you :doc:`log in to BentoCloud </bentocloud/how-tos/manage-access-token>`, use ``bentoml models push`` and ``bentoml models pull`` to upload your models to and download them from BentoCloud:

        .. code-block:: bash

            $ bentoml models push summarization-model:latest

            Successfully pushed model "summarization-model:btwtmvu5kwqc67i3"                                                                                                                                                                                           │

        .. code-block:: bash

            $ bentoml models pull summarization-model:latest

            Successfully pulled model "summarization-model:btwtmvu5kwqc67i3"

    .. tab-item:: Delete

        .. code-block:: bash

            $ bentoml models delete summarization-model:latest -y

            INFO [cli] Model(tag="summarization-model:btwtmvu5kwqc67i3") deleted

.. tip::

    Learn more about CLI usage by running ``bentoml models --help``.

Python APIs
^^^^^^^^^^^

In addition to the CLI commands, BentoML also provides equivalent Python APIs for managing models.

.. tab-set::

    .. tab-item:: List

        ``bentoml.models.list`` returns a list of ``bentoml.Model`` instances:

        .. code-block:: python

            import bentoml
            models = bentoml.models.list()

    .. tab-item:: Import/Export

        .. code-block:: python

            import bentoml
            bentoml.models.export_model('iris_clf:latest', '/path/to/folder/my_model.bentomodel')

        .. code-block:: python

            bentoml.models.import_model('/path/to/folder/my_model.bentomodel')

        You can export models to and import models from external storage devices, such as AWS S3, GCS, FTP and Dropbox. For example:

        .. code-block:: python

            bentoml.models.import_model('s3://my_bucket/folder/my_model.bentomodel')

    .. tab-item:: Push/Pull

        If you :doc:`have access to BentoCloud </bentocloud/how-tos/manage-access-token>`, you can also push local models to or pull models from it.

        .. code-block:: python

            import bentoml
            bentoml.models.push("summarization-model:latest")

        .. code-block:: python

            bentoml.models.pull("summarization-model:latest")

    .. tab-item:: Delete

        .. code-block:: python

            import bentoml
            bentoml.models.delete("summarization-model:latest")
