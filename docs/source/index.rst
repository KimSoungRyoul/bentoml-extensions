================================
BentoML-Extension (bentomlx)
================================

|github_stars| |pypi_status| |actions_status| |documentation_status| |join_slack|

----

`BentoML-Extensions <https://github.com/bentoml/BentoML>`_ A.K.A (bentomlx) provide Two additional Components,

* Intel Optimized interService(or Runner)
* FeatureStore.

``pip install bentoml bentomlx``


Featured use cases
------------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/guides/feature-store`
        :link: /guides/feature-store
        :link-type: doc

        ``pip install "bentomlx[FEATURESTORE]"``

        ex: ``pip install "bentomlx[redis]"``

        * redis
        * aerospike
        * elasticsearch

    .. grid-item-card:: :doc:`/guides/intel-optimized-inter-service`
        :link: /guides/intel-optimized-inter-service
        :link-type: doc

        ``pip install torch --index-url https://download.pytorch.org/whl/cpu``

        ``pip install "intel-extension-for-pytorch bentomlx"``

        * ipex(intel-extension-for-pytorch)
        * custom c binding model which build with oneDNN

BentoML-Extensions's Goal
--------------------------

todo: KR -> ENG

최근 LLM의 발전이 가속화됨에 따라 모델 서빙프레임워크에서도 대규모 연산량을 위해 Nvidia GPU 기반 기능들을 지원하고 개선해나가고있다.
BentoML 또한 최근 트랜드에 맞게 nvidia GPU resource호환,vLLM과 같은 고성능 inference 프레임워크 연동과 같이
GPU 모델서빙 관련 기능적인 개선에 힘을 쓰고있다.
최근 주목받고있는 모델들, 즉 Diffusion 또는 LLM계열 모델들의 연산량을 감당하기 위해서는 GPU 특히 cuda에 관한 지원은 우선순위가 높아야만한다.

이로 인해 대부분의 inference engine or serving 오픈소스들은 CPU 관련 기능들에 대한 지원이 빈약할 수 밖에 없다.
그러나 CPU inference 연산량 최적화에 대한 지원이 부족하다는 의미는 최우선순위가 아닐뿐이지 최하위우선순위라는 의미는 아니다.
is not high priority, it does not mean lowest priority.
BentoML은 매우 뛰어난 모델서빙 프레임워크다. 기존 ML프레임워크들을 통합하면서 이를 쉽게 빌드 배포할수 있도록 기능들을 제공한다.
BentoML 또한





The BentoML documentation provides detailed guidance on the project with hands-on tutorials and examples. If you are a first-time user of BentoML, we recommend that you read the following documents in order:

.. toctree::
   :caption: bentomlx
   :hidden:

   get-started/index
   use-cases/index
   guides/index
   Examples <https://github.com/bentoml/BentoML/tree/main/examples>


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
