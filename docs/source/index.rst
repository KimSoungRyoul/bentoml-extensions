================================
BentoML-Extension (bentomlx)
================================

|github_stars| |pypi_status| |actions_status| |documentation_status| |join_slack|

----

`BentoML-Extensions <https://github.com/bentoml/BentoML>`_ provide Two additional Components, Intel Optimized Runner and FeatureStore.

```
pip install bentoml bentomlx
```


Featured use cases
------------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/use-cases/featurestore`
        :link: /use-cases/featurestore
        :link-type: doc

        FeatureStore
          * redis
          * aerospike
          * elasticsearch

    .. grid-item-card:: :doc:`/use-cases/intel-optimized-runner`
        :link: /use-cases/intel-optimized-runner
        :link-type: doc

        Intel Optimized Runner
          * ipex(intel-extension-for-pytorch)
          * custom c binding model which build with oneDNN

BentoML-Extensions's Goal
--------------------------

BentoML은 매우 심플하게 엠엘프레임워크를 통합해주는 모델서빙 프레임워크이다.

The BentoML documentation provides detailed guidance on the project with hands-on tutorials and examples. If you are a first-time user of BentoML, we recommend that you read the following documents in order:

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`Get started <get-started/index>`
        :link: get-started/index
        :link-type: doc

        Gain a basic understanding of the BentoML open-source framework, its workflow, installation, and a quickstart example.

    .. grid-item-card:: :doc:`Use cases <use-cases/index>`
        :link: use-cases/index
        :link-type: doc

        Create different BentoML projects for common machine learning scenarios, like large language models, image generation, embeddings, speech recognition, and more.

    .. grid-item-card:: :doc:`Guides <guides/index>`
        :link: guides/index
        :link-type: doc

        Dive into BentoML's features and advanced use cases, including GPU support, clients, monitoring, and performance optimization.



.. |pypi_status| image:: https://img.shields.io/pypi/v/bentoml.svg?style=flat-square
   :target: https://pypi.org/project/BentoML
.. |actions_status| image:: https://github.com/bentoml/bentoml/workflows/CI/badge.svg
   :target: https://github.com/bentoml/bentoml/actions
.. |documentation_status| image:: https://readthedocs.org/projects/bentoml/badge/?version=latest&style=flat-square
   :target: https://docs.bentoml.com/
.. |join_slack| image:: https://badgen.net/badge/Join/Community%20Slack/cyan?icon=slack&style=flat-square
   :target: https://l.bentoml.com/join-slack
.. |github_stars| image:: https://img.shields.io/github/stars/bentoml/BentoML?color=%23c9378a&label=github&logo=github&style=flat-square
   :target: https://github.com/bentoml/bentoml
