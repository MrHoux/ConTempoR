'''
The benchmark automatically construction pipeline mainly includes three stages:
Stage 1: retrieve year pages: retrieve year pages for getting started with
Stage 2: construct pseudo-questions: include (i) topic entity sampling, (ii) information snippet retrieval and (iii) pseudo-question construction
Stage 3: rephrase pseudo-questions: question rephrasing via InstructGPT
'''

import json
import os
import pickle
import runpy
import subprocess
import sys
import time
from pathlib import Path

from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient

from ConTempoR.information_snippet_retrieval.wp_retriever.wikipedia_entity_retriever import WikipediaEntityPageRetriever
from ConTempoR.library.utils import get_config, get_logger, get_qid, split_time_range, target_question_for_each_range
from ConTempoR.pseudo_question_construction.pseudo_question_generation import PseudoQuestionGeneration
from ConTempoR.year_page_retrieval.year_page_retriever import YearPageRetrieval

EVENT_PAGE_PREFIX = "Portal:Current_events"


class Pipeline:
    def __init__(self, config):
        """Initialize the year range for getting started with,
        generate year/month page urls,
        split year/month pages urls into groups in which each group span a time interval such as 50 years,
        and other configurations."""

        # load config
        self.config = config
        self.logger = get_logger(__name__, config)
        self.repo_root = Path(__file__).resolve().parents[1]
        self.question_process_dir = Path(__file__).resolve().parent / "question_process"
        self._stage0_class = None

        # define months in each year
        self.months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                       "November", "December"]

        # load the start year and end year. It defines the whole time span for retrieval information/events.
        self.year_start = self.config["year_start"]
        self.year_end = self.config["year_end"]

        # load the data path and result path
        self.data_path = self.config["data_path"]
        self.result_path = self.config["result_path"]

        # load wikidata qid and wikipedia url mapping dictionary for generating year/month qid and urls
        with open(os.path.join(self.config["data_path"], self.config["path_to_wikidata_mappings"]), "rb") as fp:
            self.wikidata_mappings = pickle.load(fp)
        with open(os.path.join(self.config["data_path"], self.config["path_to_wikipedia_mappings"]), "rb") as fp:
            self.wikipedia_mappings = pickle.load(fp)

        # load year pages storing path
        year_page_output_path = os.path.join(self.result_path, f"{self.year_start}_{self.year_end}_year_page")
        self.year_page_out_dir = Path(year_page_output_path)
        self.year_page_out_dir.mkdir(parents=True, exist_ok=True)

        # generate (or load if there already is) year/month qids and urls
        year_page_qid_url_file = os.path.join(self.result_path,
                                              f"{self.year_start}_{self.year_end}_year_page_qid.pickle")
        if os.path.exists(year_page_qid_url_file):
            with open(year_page_qid_url_file, "rb") as fyear:
                self.get_year_month_page_link_pool = pickle.load(fyear)

        else:
            # generate year/month page qid and url mappings according to the start and end year range configuration
            # beyond the year pages, we go to the granularity of months, e.g.
            # https://en.wikipedia.org/wiki/Portal:Current_events/January_2022
            # or https://en.wikipedia.org/wiki/March_2022
            self.get_year_month_page_link_pool = self._year_page_pool(self.year_start, self.year_end)

            with open(year_page_qid_url_file, "wb") as fyear:
                pickle.dump(self.get_year_month_page_link_pool, fyear)

        self.logger.info(f"total number of target year range: {len(self.get_year_month_page_link_pool)}")
        self.logger.info(f"target years: {self.get_year_month_page_link_pool.keys()}")
        self.year_pages_num = sum([len(pages) for pages in self.get_year_month_page_link_pool.values()])
        self.logger.info(f"total number of year and month pages: {self.year_pages_num}")

        # the target number of pseudo-questions for generating in total
        self.target_question_total_number = self.config["target_question_number"]
        # set the sample size of entities, 150 as default.
        self.sample_size = self.config["sample_size"]
        # set year range interval, 50 years as default.
        # the year range is split into groups with time range interval.
        # the pseudo-question construction pipeline is within the time range interval so that the entity pool is not too large.
        self.year_range_interval = self.config["year_range_interval"]
        self.year_range_list = split_time_range(self.year_start, self.year_end, self.year_range_interval)
        # the total target number of questions is distributed in each group of the time range interval
        # if the total target number of questions is less than the total number of pages, we use the total number of pages to replace the total target number of questions
        self.target_question_number_per_range, self.pages_for_each_range = target_question_for_each_range(
            self.target_question_total_number, self.year_range_list, self.get_year_month_page_link_pool)

        self.logger.info(f"year range: {self.year_range_list}")
        self.logger.info(f"target questions number for each year range: {self.target_question_number_per_range}")

        # create the folder for storing the intermediate results
        output_path = os.path.join(self.result_path,
                                   f"{self.year_start}_{self.year_end}_{self.target_question_total_number}_{self.sample_size}")
        self.output_dir = Path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # topic entities
        self.topic_entities_file_path = os.path.join(self.output_dir, self.config["topic_entity_in_total_file"])
        self.pseudo_questions_file_path = os.path.join(self.output_dir, self.config["pseudo_questions_in_total_file"])

        self.topic_entities_in_total = []
        # for avoiding the duplicated topic entities, we store the topic entities and initialize via reading the file.
        # after generating the pseudo-questions, the file will be updated.
        if os.path.exists(self.topic_entities_file_path):
            with open(self.topic_entities_file_path, "r") as fp:
                for item in fp.readlines():
                    self.topic_entities_in_total.append(item.strip())

        # initialize the generated pseudo-questions
        self.pseudo_question_in_total = {}

        if config["clocq_use_api"]:
            self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
        else:
            self.clocq = CLOCQ()

        # instantiate wikipedia retriever
        self.wp_retriever = WikipediaEntityPageRetriever(config, self.clocq, self.wikidata_mappings,
                                                         self.wikipedia_mappings)

    def benchmark_construction(self):
        # step 1:
        self.year_page_retrieval()
        # step 2:
        self.pseudo_question_pipeline()
        # step 3:
        self.question_rephrase()

    # stage 1: retrieve all year and month pages for the start and end year range
    def year_page_retrieval(self):
        retrieval = YearPageRetrieval(self.config, self.year_page_out_dir, self.wp_retriever,
                                      self.get_year_month_page_link_pool)
        retrieval.retrieve_page_per_year()
        self.wp_retriever.store_dump()
        self.wp_retriever.annotator.store_cache()

    # stage 2: pipeline for generating pseudo-questions, include:
    # (i) topic entity sampling, (ii) information snippet retrieval and (iii) pseudo-question construction
    def pseudo_question_pipeline(self):
        start_total = time.time()
        # start pipeline for each year range interval. In each interval, repeat the three sub-steps:
        # (i) topic entity sampling, (ii) information snippet retrieval and (iii) pseudo-question construction
        for range in self.year_range_list:
            start = time.time()
            year_start = range[0]
            year_end = range[1]
            self.logger.info(f"Start to generate pseudo-questions for the year range: {range}")
            target_question_number = self.target_question_number_per_range[range]
            self.logger.info(f"The target question number for the year range {range} is: {target_question_number}")
            pseudo_ques_pipeline = PseudoQuestionGeneration(self.config, self.wp_retriever, self.year_page_out_dir,
                                                            year_start, year_end, self.output_dir,
                                                            self.topic_entities_in_total, target_question_number)
            pseudo_ques_pipeline.question_generate_iterative()
            self.pseudo_question_in_total.update(pseudo_ques_pipeline.pseudo_questions)
            self.topic_entities_in_total += pseudo_ques_pipeline.topic_entities
            print("Year start for this year range:", year_start)
            print("Time consumed for this year range:", time.time() - start)

        print("Total time consumed:", time.time() - start_total)
        print("Total number of topic entities:", len(self.pseudo_question_in_total.keys()))
        print("Total number of pseudo questions:",
              sum([len(self.pseudo_question_in_total[topic_entity]) for topic_entity in self.pseudo_question_in_total]))
        with open(self.topic_entities_file_path, "w") as fp:
            for item in list(self.pseudo_question_in_total.keys()):
                fp.write(item)
                fp.write("\n")

        with open(self.pseudo_questions_file_path, "w") as fout:
            fout.write(json.dumps(self.pseudo_question_in_total, indent=4))

        self.wp_retriever.store_dump()
        self.wp_retriever.annotator.store_cache()

    def question_rephrase(self):
        """Run the question rephrase stage."""
        workflow_cfg = self.config.get("question_rephrase_workflow")
        if workflow_cfg and workflow_cfg.get("enabled", True):
            self._run_question_rephrase_workflow(workflow_cfg)
            return

        raise RuntimeError(
            "question_rephrase_workflow is disabled or missing; please enable it in the config."
        )

    def _resolve_path(self, value):
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return path

    def _run_command(self, cmd, cwd=None):
        self.logger.info(f"Running command: {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=cwd or self.repo_root)
        if completed.returncode != 0:
            raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")

    def _load_stage0_class(self):
        if self._stage0_class is not None:
            return self._stage0_class
        script_path = self.question_process_dir / "sample_pseudo_question_for_rephrase.py"
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing stage0 script: {script_path}")
        module_globals = runpy.run_path(str(script_path))
        cls = module_globals.get("PseudoQuestionSampleRephrase")
        if cls is None:
            raise RuntimeError("PseudoQuestionSampleRephrase not found in stage0 script.")
        self._stage0_class = cls
        return self._stage0_class

    def _run_stage0_merge(self, stage0_cfg: dict) -> Path:
        stage0_cfg = stage0_cfg or {}
        output_path = self._resolve_path(
            stage0_cfg.get("output_path", "_intermediate_results/data_processing/00_initial_questions.json")
        )
        if output_path is None:
            raise ValueError("Stage0 requires an `output_path`.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sampler_cls = self._load_stage0_class()
        sampler = sampler_cls(self.config)
        results = sampler.run_merge_filter_and_format()
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        return output_path

    def _run_question_rephrase_workflow(self, workflow_cfg):
        stage0_output = self._run_stage0_merge(workflow_cfg.get("stage0", {}))
        self.logger.info(f"[Stage 0] merged pseudo-questions -> {stage0_output}")

        data_filter_cfg = workflow_cfg.get("data_filter", {})
        filtered_output = stage0_output
        if data_filter_cfg.get("enabled", True):
            filtered_output = self._run_data_filter_stage(data_filter_cfg, stage0_output)
            self.logger.info(f"[Stage 1] data_filter output -> {filtered_output}")

        rephrase_cfg = workflow_cfg.get("rephrase_stage1", {})
        rephrase_output = self._run_stage1_rephrase(rephrase_cfg, filtered_output)
        self.logger.info(f"[Stage 2] question rephraser output -> {rephrase_output}")

        pipeline_cfg = workflow_cfg.get("pipeline_2_7", {})
        stage5_output = self._run_stage_2_7_pipeline(pipeline_cfg, rephrase_output)
        self.logger.info(f"[Stage 3] 2_7 pipeline output -> {stage5_output}")

        stage6_cfg = workflow_cfg.get("stage6", {})
        final_output = self._run_stage6_connect(stage6_cfg, stage5_output)
        self.logger.info(f"[Stage 4] question connect output -> {final_output}")

    def _run_data_filter_stage(self, cfg: dict, default_input: Path) -> Path:
        script_path = self.question_process_dir / "data_filter.py"
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing data filter script: {script_path}")

        cfg = cfg or {}
        input_override = cfg.get("input_path")
        input_path = self._resolve_path(input_override) if input_override else default_input
        if input_path is None:
            raise ValueError("Data filter stage requires an input path.")
        if not input_path.exists():
            raise FileNotFoundError(f"Data filter input not found: {input_path}")

        output_path = self._resolve_path(cfg.get("output_path"))
        if output_path is None:
            raise ValueError("Data filter stage requires `output_path`.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]

        grouped_topic_out = self._resolve_path(cfg.get("grouped_topic_output"))
        if grouped_topic_out:
            grouped_topic_out.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--grouped-topic-output", str(grouped_topic_out)]

        pqc_output = self._resolve_path(cfg.get("pqc_output"))
        if pqc_output:
            pqc_output.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--pqc-output", str(pqc_output)]

        intermediate_dir = self._resolve_path(cfg.get("intermediate_dir"))
        if intermediate_dir:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--intermediate-dir", str(intermediate_dir)]

        if cfg.get("seed") is not None:
            cmd += ["--seed", str(cfg.get("seed"))]
        if cfg.get("extract_min_len") is not None:
            cmd += ["--extract-min-len", str(cfg.get("extract_min_len"))]
        if cfg.get("extract_dedup"):
            cmd.append("--extract-dedup")

        self._run_command(cmd)
        return output_path

    def _run_stage1_rephrase(self, cfg: dict, default_input: Path) -> Path:
        script_path = self.question_process_dir / "question_rephraser.py"
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing rephrase script: {script_path}")

        cfg = cfg or {}
        input_override = cfg.get("input_path")
        input_path = self._resolve_path(input_override) if input_override else default_input
        if input_path is None:
            raise ValueError("Stage1 rephrase requires an input path.")
        if not input_path.exists():
            raise FileNotFoundError(f"Stage1 input not found: {input_path}")

        output_path = self._resolve_path(cfg.get("output_path"))
        if output_path is None:
            raise ValueError("Stage1 rephrase requires `output_path`.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rephrase_cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(cfg.get("model", self.config.get("gpt3_model", "gpt-3.5-turbo"))),
            "--temperature",
            str(cfg.get("temperature", 0.0)),
            "--max-tokens",
            str(cfg.get("max_tokens", 256)),
        ]
        api_key = cfg.get("api_key", "")
        if api_key:
            rephrase_cmd += ["--api-key", api_key]
        organization = cfg.get("organization", "")
        if organization:
            rephrase_cmd += ["--organization", organization]
        if cfg.get("use_cache"):
            rephrase_cmd.append("--use-cache")
            cache_dir = self._resolve_path(cfg.get("cache_dir"))
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                rephrase_cmd += ["--cache-dir", str(cache_dir)]
        if cfg.get("overwrite"):
            rephrase_cmd.append("--overwrite")
        if cfg.get("signal_retry"):
            rephrase_cmd.append("--signal-retry")
        rephrase_cmd += ["--signal-retry-max", str(cfg.get("signal_retry_max", 2))]

        self._run_command(rephrase_cmd)
        return output_path

    def _run_stage_2_7_pipeline(self, cfg: dict, default_input: Path) -> Path:
        script_path = self.question_process_dir / "pipeline.py"
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing 2_7 pipeline script: {script_path}")

        cfg = cfg or {}
        input_override = cfg.get("input_path")
        input_path = self._resolve_path(input_override) if input_override else default_input
        if input_path is None:
            raise ValueError("2_7 pipeline requires an input path.")
        if not input_path.exists():
            raise FileNotFoundError(f"2_7 pipeline input not found: {input_path}")

        output_path = self._resolve_path(cfg.get("output_path"))
        if output_path is None:
            raise ValueError("2_7 pipeline requires `output_path`.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline_cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        intermediate_dir = self._resolve_path(cfg.get("intermediate_dir"))
        if intermediate_dir:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            pipeline_cmd += ["--intermediate-dir", str(intermediate_dir)]
        stop_stage = cfg.get("stop_after_stage")
        if stop_stage:
            pipeline_cmd += ["--stop-after-stage", str(stop_stage)]

        filter_opts = cfg.get("filter_options", {})
        pipeline_cmd += [
            "--stage2-min-dicts",
            str(filter_opts.get("stage2_min_dicts", 5)),
            "--stage3-max-question-entities",
            str(filter_opts.get("stage3_max_question_entities", 4)),
            "--stage4-min-turns",
            str(filter_opts.get("stage4_min_turns", 5)),
            "--stage5-min-length",
            str(filter_opts.get("stage5_min_length", 5)),
        ]

        stage6_proxy = cfg.get("stage6_proxy", {})
        pipeline_cmd += ["--stage6-seed", str(stage6_proxy.get("seed", 42))]
        stage6_api_key = stage6_proxy.get("api_key", "")
        if stage6_api_key:
            pipeline_cmd += ["--stage6-api-key", stage6_api_key]
        gender_model = stage6_proxy.get("gender_model")
        if gender_model:
            pipeline_cmd += ["--stage6-gender-model", gender_model]
        polish_model = stage6_proxy.get("polish_model")
        if polish_model:
            pipeline_cmd += ["--stage6-polish-model", polish_model]
        pipeline_cmd += ["--stage6-temperature", str(stage6_proxy.get("temperature", 0.0))]
        gender_cache = self._resolve_path(stage6_proxy.get("gender_cache"))
        if gender_cache:
            gender_cache.parent.mkdir(parents=True, exist_ok=True)
            pipeline_cmd += ["--stage6-gender-cache", str(gender_cache)]
        if stage6_proxy.get("disable_gpt_polish"):
            pipeline_cmd.append("--stage6-disable-gpt-polish")
        pipeline_cmd += ["--stage6-max-tries", str(stage6_proxy.get("max_tries", 9))]
        pipeline_cmd += ["--stage6-max-polish-tries", str(stage6_proxy.get("max_polish_tries", 1))]
        pipeline_cmd += ["--stage6-s1-prob", str(stage6_proxy.get("s1_prob", 0.33))]
        if stage6_proxy.get("progress"):
            pipeline_cmd.append("--stage6-progress")

        self._run_command(pipeline_cmd)
        return output_path

    def _run_stage6_connect(self, cfg: dict, default_input: Path) -> Path:
        script_path = self.question_process_dir / "question_connect.py"
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing question_connect script: {script_path}")

        cfg = cfg or {}
        input_override = cfg.get("input_path")
        input_path = self._resolve_path(input_override) if input_override else default_input
        if input_path is None:
            raise ValueError("Stage6 requires an input path.")
        if not input_path.exists():
            raise FileNotFoundError(f"Stage6 input not found: {input_path}")

        output_path = self._resolve_path(cfg.get("output_path"))
        if output_path is None:
            raise ValueError("Stage6 requires `output_path`.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--seed",
            str(cfg.get("seed", 42)),
        ]
        api_key = cfg.get("api_key", "")
        if api_key:
            cmd += ["--api-key", api_key]
        gender_model = cfg.get("gender_model")
        if gender_model:
            cmd += ["--gender-model", gender_model]
        polish_model = cfg.get("polish_model")
        if polish_model:
            cmd += ["--polish-model", polish_model]
        cmd += ["--temperature", str(cfg.get("temperature", 0.0))]
        gender_cache = self._resolve_path(cfg.get("gender_cache"))
        if gender_cache:
            gender_cache.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--gender-cache", str(gender_cache)]
        if cfg.get("disable_gpt_polish"):
            cmd.append("--disable-gpt-polish")
        cmd += ["--max-tries", str(cfg.get("max_tries", 9))]
        cmd += ["--max-polish-tries", str(cfg.get("max_polish_tries", 1))]
        cmd += ["--s1-prob", str(cfg.get("s1_prob", 0.33))]
        if cfg.get("progress"):
            cmd.append("--progress")

        self._run_command(cmd)
        return output_path

    def _year_page_pool(self, year_start, year_end):
        ''' generate year qid and url mappings, for example:

                            "1909": [
                        {
                            "id": "Q2057",
                            "wiki_path": "1909",
                            "label": "1909",
                            "page_type": "year"
                        },
                        {
                            "id": "Q6155680",
                            "wiki_path": "January_1909",
                            "label": "January 1909",
                            "page_type": "event"
                        }]
                            '''
        # map the year range to year
        year_month_pool = {year: [] for year in range(int(year_start), int(year_end) + 1)}
        for year in range(int(year_start), int(year_end) + 1):
            wiki_path = f"{year}"
            # check if year is in wikidata mapping dictionary
            if self.wikidata_mappings.get(wiki_path):
                year_wikidata_id = self.wikidata_mappings.get(wiki_path)
            elif self.wikidata_mappings.get(wiki_path.replace(".", "")):
                year_wikidata_id = self.wikidata_mappings.get(wiki_path.replace(".", ""))
            # get year qid using online service
            else:
                year_wiki_path = f"https://en.wikipedia.org/wiki/{year}"
                try:
                    year_result = get_qid(year_wiki_path)
                    if year_result:
                        year_wikidata_id = year_result[0]
                except:
                    self.logger.info(f"There is no wikidata qid for {year}")
                    continue

            year_month_pool[year].append(
                {'id': year_wikidata_id, 'wiki_path': wiki_path, 'label': str(year), 'page_type': 'year'})

            for month in self.months:
                # There are two kinds of month page url:
                # (1) month_year (2) Portal:Current_events/month_year
                # We take care of them respectively and check which one is effective
                month_wiki_path_1 = f"{month}_{year}"
                month_wiki_path_2 = f"{EVENT_PAGE_PREFIX}/{month_wiki_path_1}"
                if month_wiki_path_1 == "May_2010":
                    # There is an error in the wikidata mapping dictionary
                    # we manually add the link
                    # but this can be removed
                    year_month_pool[year].append(
                        {'id': "Q239311", 'wiki_path': month_wiki_path_2, 'label': f"{month} {year}",
                         'page_type': 'event'})
                    continue
                if self.wikidata_mappings.get(month_wiki_path_2):
                    wikidata_id = self.wikidata_mappings.get(month_wiki_path_2)
                    if wikidata_id != year_wikidata_id:
                        year_month_pool[year].append(
                            {'id': wikidata_id, 'wiki_path': month_wiki_path_2, 'label': f"{month} {year}",
                             'page_type': 'event'})
                elif self.wikidata_mappings.get(month_wiki_path_1):
                    wikidata_id = self.wikidata_mappings.get(month_wiki_path_1)
                    if wikidata_id != year_wikidata_id:
                        year_month_pool[year].append(
                            {'id': wikidata_id, 'wiki_path': month_wiki_path_1, 'label': f"{month} {year}",
                             'page_type': 'event'})
                    elif year >= 2021:
                        # when year is greater than 2021, the url changes to another format
                        wikipedia_link = f"https://en.wikipedia.org/wiki/{month_wiki_path_2}"
                        result = get_qid(wikipedia_link)
                        if result and result[0] != year_wikidata_id:
                            wikidata_id = result[0]
                            year_month_pool[year].append(
                                {'id': wikidata_id, 'wiki_path': month_wiki_path_2, 'label': f"{month} {year}",
                                 'page_type': 'event'})
                        else:
                            wikipedia_link = f"https://en.wikipedia.org/wiki/{month_wiki_path_1}"
                            result = get_qid(wikipedia_link)
                            if result and result[0] != year_wikidata_id:
                                wikidata_id = result[0]
                                year_month_pool[year].append(
                                    {'id': wikidata_id, 'wiki_path': month_wiki_path_1, 'label': f"{month} {year}",
                                     'page_type': 'event'})

        return year_month_pool


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python ConTempoR/pipeline.py <FUNCTION> <PATH_TO_CONFIG>"
        )

    # load config
    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    if function == "--year-page-retrieve":
        benchmark = Pipeline(config)
        benchmark.year_page_retrieval()

    elif function == "--pseudoquestion-generate":
        benchmark = Pipeline(config)
        benchmark.pseudo_question_pipeline()

    elif function == "--question-rephrase":
        benchmark = Pipeline(config)
        benchmark.question_rephrase()

    else:
        raise Exception(f"Unknown function {function}!")
