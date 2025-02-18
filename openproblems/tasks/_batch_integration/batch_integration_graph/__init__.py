from .... import utils
from . import api
from . import datasets
from . import methods
from . import metrics

_task_name = "Batch integration graph"
_task_summary = (
    "Removing batch effects while preserving biological variation (graph output)"
)

DATASETS = utils.get_callable_members(datasets)
METHODS = utils.get_callable_members(methods)
METRICS = utils.get_callable_members(metrics)
