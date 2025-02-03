from .tasks import get_task


class TaskManager:
    def __init__(
        self,
        meta_train_tasks,
        meta_test_tasks,
        task_setting,
    ):

        self.meta_train_tasks = meta_train_tasks
        self.meta_test_tasks = meta_test_tasks
        self.task_setting = task_setting

        self.tasks = self._get_tasks()
        self.meta_test_tasks = self._get_meta_test_tasks()

    def _prepare_task(self, task_name):
        task = get_task(task_name=task_name)(task_name=task_name, **self.task_setting)
        return task

    def _get_tasks(self):
        tasks = [self._prepare_task(task_name) for task_name in self.meta_train_tasks]
        return tasks

    def _get_meta_test_tasks(self):
        tasks = [self._prepare_task(task_name) for task_name in self.meta_test_tasks]
        return tasks
