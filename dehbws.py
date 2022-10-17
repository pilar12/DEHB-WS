import os
import sys
import time
from typing import Any, Callable, Optional

import ConfigSpace
import numpy as np
from dehb import DEHB
from distributed import Client
from loguru import logger

logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB",
}


class DEHBWS(DEHB):
    def __init__(
        self,
        cs: Optional[ConfigSpace.Configuration] = None,
        f: Optional[Callable] = None,
        dimensions: Optional[int] = None,
        mutation_factor: float = 0.5,
        crossover_prob: float = 0.5,
        strategy: str = "rand1_bin",
        min_budget: Optional[int] = None,
        max_budget: Optional[int] = None,
        eta: int = 3,
        min_clip: Optional[float] = None,
        max_clip: Optional[float] = None,
        configspace: bool = True,
        boundary_fix_type: str = "random",
        max_age=np.inf,
        n_workers: Optional[int] = None,
        client: Optional[Any] = None,
        update_func: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            cs=cs,
            f=f,
            dimensions=dimensions,
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            strategy=strategy,
            min_budget=min_budget,
            max_budget=max_budget,
            eta=eta,
            min_clip=min_clip,
            max_clip=max_clip,
            configspace=configspace,
            boundary_fix_type=boundary_fix_type,
            max_age=max_age,
            n_workers=n_workers,
            client=client,
            **kwargs
        )

        self.update_func = update_func

    def _get_next_job(self):
        """Loads a configuration and budget to be evaluated next by a free worker"""
        bracket = None
        if len(self.active_brackets) == 0 or np.all(
            [bracket.is_bracket_done() for bracket in self.active_brackets]
        ):
            # start new bracket when no pending jobs from existing brackets or empty bracket list
            bracket = self._start_new_bracket()
        else:
            for _bracket in self.active_brackets:
                # check if _bracket is not waiting for previous rung results of same bracket
                # _bracket is not waiting on the last rung results
                # these 2 checks allow DEHB to have a "synchronous" Successive Halving
                if not _bracket.previous_rung_waits() and _bracket.is_pending():
                    # bracket eligible for job scheduling
                    bracket = _bracket
                    break
            if bracket is None:
                # start new bracket when existing list has all waiting brackets
                bracket = self._start_new_bracket()
        # budget that the SH bracket allots
        budget = bracket.get_next_job_budget()
        config, parent_id = self._acquire_config(bracket, budget)
        # notifies the Bracket Manager that a single config is to run for the budget chosen
        print(self.vector_to_configspace(config))
        job_info = {
            "config": config,
            "budget": budget,
            "parent_id": parent_id,
            "bracket_id": bracket.bracket_id,
            "effective_budget": None,
        }
        if bracket.current_rung != 0:
            effective_budget = bracket.budgets[bracket.current_rung - 1]
            job_info["effective_budget"] = effective_budget
        return job_info

    def submit_job(self, job_info, **kwargs):
        """Asks a free worker to run the objective function on config and budget"""
        job_info["kwargs"] = (
            self.shared_data if self.shared_data is not None else kwargs
        )
        if job_info["effective_budget"]:
            job_info["kwargs"]["effective_budget"] = job_info["effective_budget"]
        # submit to Dask client
        if self.n_workers > 1 or isinstance(self.client, Client):
            if self.single_node_with_gpus:
                # managing GPU allocation for the job to be submitted
                job_info.update({"gpu_devices": self._get_gpu_id_with_low_load()})
            self.futures.append(self.client.submit(self._f_objective, job_info))
        else:
            # skipping scheduling to Dask worker to avoid added overheads in the synchronous case
            self.futures.append(self._f_objective(job_info))

        # pass information of job submission to Bracket Manager
        for bracket in self.active_brackets:
            if bracket.bracket_id == job_info["bracket_id"]:
                # registering is IMPORTANT for Bracket Manager to perform SH
                bracket.register_job(job_info["budget"])
                break

    def _fetch_results_from_workers(self):
        """Iterate over futures and collect results from finished workers"""
        if self.n_workers > 1 or isinstance(self.client, Client):
            done_list = [
                (i, future) for i, future in enumerate(self.futures) if future.done()
            ]
        else:
            # Dask not invoked in the synchronous case
            done_list = [(i, future) for i, future in enumerate(self.futures)]
        if len(done_list) > 0:
            self.logger.debug(
                "Collecting {} of the {} job(s) active.".format(
                    len(done_list), len(self.futures)
                )
            )
        for _, future in done_list:
            if self.n_workers > 1 or isinstance(self.client, Client):
                run_info = future.result()
                if "device_id" in run_info:
                    # updating GPU usage
                    self.gpu_usage[run_info["device_id"]] -= 1
                    self.logger.debug(
                        "GPU device released: {}".format(run_info["device_id"])
                    )
                future.release()
            else:
                # Dask not invoked in the synchronous case
                run_info = future
            # update bracket information
            fitness, cost = run_info["fitness"], run_info["cost"]
            info = run_info["info"] if "info" in run_info else dict()
            budget, parent_id = run_info["budget"], run_info["parent_id"]
            config = run_info["config"]
            bracket_id = run_info["bracket_id"]
            _bracket = None
            for bracket in self.active_brackets:
                if bracket.bracket_id == bracket_id:
                    # bracket job complete
                    bracket.complete_job(budget)  # IMPORTANT to perform synchronous SH
                    _bracket = bracket

            # carry out DE selection
            if fitness <= self.de[budget].fitness[parent_id]:
                self.de[budget].population[parent_id] = config
                self.de[budget].fitness[parent_id] = fitness
            # updating incumbents
            if self.de[budget].fitness[parent_id] < self.inc_score:
                self._update_incumbents(
                    config=self.de[budget].population[parent_id],
                    score=self.de[budget].fitness[parent_id],
                    info=info,
                )
            # bookkeeping
            self._update_trackers(
                traj=self.inc_score,
                runtime=cost,
                history=(
                    config.tolist(),
                    float(fitness),
                    float(cost),
                    float(budget),
                    info,
                ),
            )
            if (
                _bracket.is_bracket_done()
                or self._is_run_budget_exhausted(self.fevals)
                or budget != _bracket.get_budget()
            ):
                if (
                    not _bracket._is_rung_waiting(_bracket.current_rung - 1)
                    and not _bracket._is_rung_pending(_bracket.current_rung - 1)
                ) or self._is_run_budget_exhausted(self.fevals):
                    self.logger.info("Updating Supernet")
                    self.update_func(
                        [
                            self.de[budget].vector_to_configspace(config)
                            for config in self.de[budget].population
                        ]
                    )
        # remove processed future
        self.futures = np.delete(self.futures, [i for i, _ in done_list]).tolist()

    @logger.catch
    def run(
        self,
        fevals=None,
        brackets=None,
        total_cost=None,
        single_node_with_gpus=False,
        verbose=False,
        debug=False,
        save_intermediate=True,
        save_history=True,
        name=None,
        **kwargs
    ):
        """Main interface to run optimization by DEHB

        This function waits on workers and if a worker is free, asks for a configuration and a
        budget to evaluate on and submits it to the worker. In each loop, it checks if a job
        is complete, fetches the results, carries the necessary processing of it asynchronously
        to the worker computations.

        The duration of the DEHB run can be controlled by specifying one of 3 parameters. If more
        than one are specified, DEHB selects only one in the priority order (high to low):
        1) Number of function evaluations (fevals)
        2) Number of Successive Halving brackets run under Hyperband (brackets)
        3) Total computational cost (in seconds) aggregated by all function evaluations (total_cost)
        """
        # checks if a Dask client exists
        if len(kwargs) > 0 and self.n_workers > 1 and isinstance(self.client, Client):
            # broadcasts all additional data passed as **kwargs to all client workers
            # this reduces overload in the client-worker communication by not having to
            # serialize the redundant data used by all workers for every job
            self.shared_data = self.client.scatter(kwargs, broadcast=True)

        # allows each worker to be mapped to a different GPU when running on a single node
        # where all available GPUs are accessible
        self.single_node_with_gpus = single_node_with_gpus
        if self.single_node_with_gpus:
            self.distribute_gpus()

        self.fevals = fevals
        self.brackets = brackets
        self.total_cost = total_cost

        self.start = time.time()
        if verbose:
            print(
                "\nLogging at {} for optimization starting at {}\n".format(
                    os.path.join(os.getcwd(), self.log_filename),
                    time.strftime("%x %X %Z", time.localtime(self.start)),
                )
            )
        if debug:
            logger.configure(handlers=[{"sink": sys.stdout}])
        while True:
            if self._is_run_budget_exhausted(fevals, brackets, total_cost):
                break
            if self.is_worker_available():
                job_info = self._get_next_job()
                if brackets is not None and job_info["bracket_id"] >= brackets:
                    # ignore submission and only collect results
                    # when brackets are chosen as run budget, an extra bracket is created
                    # since iteration_counter is incremented in _get_next_job() and then checked
                    # in _is_run_budget_exhausted(), therefore, need to skip suggestions
                    # coming from the extra allocated bracket
                    # _is_run_budget_exhausted() will not return True until all the lower brackets
                    # have finished computation and returned its results
                    pass
                else:
                    if self.n_workers > 1 or isinstance(self.client, Client):
                        self.logger.debug(
                            "{}/{} worker(s) available.".format(
                                self._get_worker_count() - len(self.futures),
                                self._get_worker_count(),
                            )
                        )
                    # submits job_info to a worker for execution
                    self.submit_job(job_info, **kwargs)
                    if verbose:
                        budget = job_info["budget"]
                        self._verbosity_runtime(fevals, brackets, total_cost)
                        self.logger.info(
                            "Evaluating a configuration with budget {} under "
                            "bracket ID {}".format(budget, job_info["bracket_id"])
                        )
                        self.logger.info(
                            "Best score seen/Incumbent score: {}".format(self.inc_score)
                        )
                    self._verbosity_debug()
            self._fetch_results_from_workers()
            if save_intermediate and self.inc_config is not None:
                self._save_incumbent(name)
            if save_history and self.history is not None:
                self._save_history(name)
            self.clean_inactive_brackets()
        # end of while

        if verbose and len(self.futures) > 0:
            self.logger.info(
                "DEHB optimisation over! Waiting to collect results from workers running..."
            )
        while len(self.futures) > 0:
            self._fetch_results_from_workers()
            if save_intermediate and self.inc_config is not None:
                self._save_incumbent(name)
            if save_history and self.history is not None:
                self._save_history(name)
            time.sleep(0.05)  # waiting 50ms

        if verbose:
            time_taken = time.time() - self.start
            self.logger.info(
                "End of optimisation! Total duration: {}; Total fevals: {}\n".format(
                    time_taken, len(self.traj)
                )
            )
            self.logger.info("Incumbent score: {}".format(self.inc_score))
            self.logger.info("Incumbent config: ")
            if self.configspace:
                config = self.vector_to_configspace(self.inc_config)
                for k, v in config.get_dictionary().items():
                    self.logger.info("{}: {}".format(k, v))
            else:
                self.logger.info("{}".format(self.inc_config))
        self._save_incumbent(name)
        self._save_history(name)
        return (
            np.array(self.traj),
            np.array(self.runtime),
            np.array(self.history, dtype=object),
        )
