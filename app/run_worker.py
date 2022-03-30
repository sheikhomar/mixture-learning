import os, subprocess, json, shutil, time

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import click
import psutil


class JobInfo:
    process_id: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    working_dir: Path
    command: str
    command_params: Dict[str, object]

    def __init__(self) -> None:
        self.process_id = -1
        self.started_at = None
        self.completed_at = None
        self.working_dir = None
        self.command = ""
        self.command_params = dict()

    @classmethod
    def load_json(cls, job_info_path: Path):
        def parse_datetime(date_str: Optional[str]):
            if date_str is not None and len(date_str) > 0:
                return datetime.fromisoformat(date_str)
            return None

        with open(job_info_path, "r") as f:
            content = json.load(f)
            obj = cls()
            obj.process_id = content["process_id"]
            obj.started_at = parse_datetime(content["started_at"])
            obj.completed_at = parse_datetime(content["completed_at"])
            obj.working_dir = Path(content["working_dir"])
            obj.command = content["command"]
            obj.command_params = content["command_params"]
            return obj

    def write_json(self, job_info_path: Path) -> None:
        def datetime_to_iso_str(dt: Optional[datetime]):
            if dt is None:
                return None
            return dt.isoformat()

        job_dict = {
            "process_id": self.process_id,
            "started_at": datetime_to_iso_str(self.started_at),
            "completed_at": datetime_to_iso_str(self.completed_at),
            "working_dir": str(self.working_dir),
            "command": self.command,
            "command_params": self.command_params,
        }
        with open(job_info_path, "w") as f:
            json.dump(job_dict, f, indent=4, sort_keys=False)
    
    def get_process_command(self) -> List[str]:
        cmd = self.command.split(" ")
        for param_name, param_value in self.command_params.items():
            cmd.append(f"--{param_name}")
            cmd.append(f"{param_value}")
        return cmd

class Worker:
    def __init__(self, queue_dir: str, max_active: int, n_threads: int) -> None:
        self._queue_dir = queue_dir
        self._max_active = max_active
        self._dir_ready = os.path.join(queue_dir, "ready")
        self._dir_completed = os.path.join(queue_dir, "completed")
        self._dir_in_progress = os.path.join(queue_dir, "in-progress")
        self._dir_discarded = os.path.join(queue_dir, "discarded")
        self._n_threads = n_threads

        self._child_processes: List[subprocess.Popen] = []

        for directory in [self._dir_ready, self._dir_in_progress, self._dir_completed, self._dir_discarded]:
            if not os.path.exists(directory):
                print(f"Creating directory {directory}...")
                os.makedirs(directory)

    def run(self) -> None:
        prev_has_launched = True
        has_launched = True
        while True:
            self._clean_in_progress()
            n_active = len(self._find_in_progress_files())
            if n_active < self._max_active:
                has_launched = self._lunch_new_run()
                if not has_launched and prev_has_launched:
                    print("No more jobs in queue.")
            time.sleep(2)
            prev_has_launched = has_launched

    def _clean_in_progress(self):
        job_info_paths = self._find_in_progress_files()
        for job_info_path in job_info_paths:
            job = JobInfo.load_json(job_info_path)
            if not self._is_running(job.process_id):
                print(f"Process {job.process_id} which started {job.started_at} is not running anymore.")

                # Check if the result file is created.
                done_job_info_path = job.working_dir / "done.out"
                if done_job_info_path.exists():
                    completed_at = datetime.fromtimestamp(done_job_info_path.stat().st_ctime)
                    print(f" - Completed at {completed_at}. Moving to completed.")
                    job.completed_at = completed_at
                    job.process_id = -2
                    job.write_json(job_info_path)
                    shutil.move(job_info_path, f"{self._dir_completed}/{job_info_path.name}")
                    shutil.copy(
                        src=f"{self._dir_completed}/{job_info_path.name}", 
                        dst=f"{job.working_dir}/job-info.json"
                    )
                else:
                    print(" - Process stopped but done.out file does not exist! Discarding run.")
                    self._move_to_discarded(job_info_path)

    def _is_running(self, process_id: int) -> bool:
        for proc in self._child_processes:
            if proc.pid == process_id:
                # poll() checks the process has terminated. Returns None value 
                # if process has not terminated yet.
                return proc.poll() is None

        # psutil cannot detect child processes.
        for proc in psutil.process_iter():
            if process_id == proc.pid:
                return True
        return False

    def _find_json_files(self, dir_name: str) -> List[Path]:
        dir_name_str = str(dir_name)
        return [
            Path(f"{dir_name_str}/{file_name}")
            for file_name in os.listdir(dir_name_str)
            if file_name.endswith(".json")
        ]

    def _find_in_progress_files(self) -> List[Path]:
        return self._find_json_files(self._dir_in_progress)

    def _should_discard(self, job_info_path: Path) -> bool:
        paths_to_check = [
            os.path.join(self._dir_in_progress, job_info_path.name),
            os.path.join(self._dir_completed  , job_info_path.name),
        ]
        for p in paths_to_check:
            if os.path.exists(p):
                return True
        return False

    def _move_to_discarded(self, job_info_path: Path) -> None:
        discarded_file_name = job_info_path.name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        discarded_path = f"{self._dir_discarded}/{discarded_file_name}"
        shutil.move(job_info_path, discarded_path)
        print(f"Discarding job {job_info_path.name}. Moving to {discarded_path}...")

    def _get_next_job_info_path(self) -> Optional[Path]:
        file_paths = self._find_json_files(self._dir_ready)
        if len(file_paths) > 0:
            sorted_file_paths = list(sorted(file_paths, key=lambda file_path: file_path.name))
            return sorted_file_paths[0]
        return None

    def _move_to_progress(self, job_info_path: Path) -> Path:
        in_progress_path = f"{self._dir_in_progress}/{job_info_path.name}"
        shutil.move(job_info_path, in_progress_path)
        return in_progress_path

    def _lunch_new_run(self) -> False:
        # Find the next run file containing the experiment to run
        while True:
            job_info_path = self._get_next_job_info_path()
            if job_info_path is None:
                return False
            if self._should_discard(job_info_path):
                print(f"Found job that is already running or completed.")
                self._move_to_discarded(job_info_path)
            else:
                print(f"Will run job with config {job_info_path}")
                break

        # Prepare for launch
        job_info_path = self._move_to_progress(job_info_path)
        job_info: JobInfo = JobInfo.load_json(job_info_path)
        cmd = job_info.get_process_command()
        cmd += ["--n-threads", f"{self._n_threads}"]

        # Actual launch
        print(f"Launching experiment with command:\n '{cmd}'")
        p = subprocess.Popen(
            args=cmd,
            stdout=open(job_info.working_dir / "stdout.txt", "a"),
            stderr=open(job_info.working_dir / "stderr.txt", "a"),
            start_new_session=True
        )
        self._child_processes.append(p)

        # Create PID file.
        with open(job_info.working_dir / "pid.txt", "w") as f:
            f.write(str(p.pid))

        # Persist run details to disk.
        job_info.started_at = datetime.now()
        job_info.process_id = p.pid
        job_info.write_json(job_info_path)

        return True

@click.command(help="Runs jobs.")
@click.option(
    "-q",
    "--queue-dir",
    type=click.STRING,
    required=True,
)
@click.option(
    "-m",
    "--max-active",
    type=click.INT,
    default=1,
    help="Maximum number of simultaneous jobs."
)
@click.option(
    "--n-threads",
    type=click.INT,
    default=1,
    help="Maximum number of threads per job."
)
def main(queue_dir: str, max_active: int, n_threads: int):
    Worker(
        queue_dir=queue_dir,
        max_active=max_active,
        n_threads=n_threads,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
