"""MLCube handler file"""
import os
import subprocess
import typer
import yaml
import shutil

typer_app = typer.Typer()

class DownloadTask:
    """Download dataset"""

    @staticmethod
    def run(parameters_file: str, output_path: str) -> None:

        cmd = "python3 download_data.py"
        cmd += f" --parameters_file={parameters_file} --output_path={output_path}"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()


class CreateBaselinesTask:
    """Run create_baselines script"""

    @staticmethod
    def run(
        embedding_folder: str, groundtruth_folder: str, submission_folder: str
    ) -> None:
        os.symlink(embedding_folder, "embeddings")
        os.symlink(groundtruth_folder, "data")
        shutil.rmtree("submissions", ignore_errors=True)
        os.symlink(submission_folder, "submissions")
        cmd = "python3 create_baselines.py"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()

class EvaluateTask:
    """Execute evaluation script"""

    @staticmethod
    def run(
        submission_folder: str,
        groundtruth_folder: str,
        embedding_folder: str,
        results_folder: str,
    ) -> None:
        # ignore errors should only happen when running for the first time
        # when there is no submissions folder
        shutil.rmtree("submissions", ignore_errors=True)
        os.symlink(embedding_folder, "embeddings")
        os.symlink(submission_folder, "submissions")
        os.symlink(groundtruth_folder, "data")
        os.symlink(results_folder, "results")
        cmd = "python3 main.py"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()


class PlotTask:
    """Execute plot script"""

    @staticmethod
    def run(results_folder: str, submission_folder: str) -> None:

        os.symlink(results_folder, "results")
        os.symlink(submission_folder, "submissions")
        cmd = "python3 plotter.py"
        splitted_cmd = cmd.split()
        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()

        cmd = "python3 plotter_speed_2.py"
        splitted_cmd = cmd.split()
        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()


@typer_app.command("download")
def download(
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_path: str = typer.Option(..., "--output_path"),
):
    DownloadTask.run(parameters_file, output_path)


@typer_app.command("create_baselines")
def create_baselines(
    embedding_folder: str = typer.Option(..., "--embedding_folder"),
    groundtruth_folder: str = typer.Option(..., "--groundtruth_folder"),
    submission_folder: str = typer.Option(..., "--submission_folder"),
):
    CreateBaselinesTask.run(embedding_folder, groundtruth_folder, submission_folder)


@typer_app.command("evaluate")
def evaluate(
    submission_folder: str = typer.Option(..., "--submission_folder"),
    groundtruth_folder: str = typer.Option(..., "--groundtruth_folder"),
    embedding_folder: str = typer.Option(..., "--embedding_folder"),
    results_folder: str = typer.Option(..., "--results_folder"),
):
    EvaluateTask.run(
        submission_folder, groundtruth_folder, embedding_folder, results_folder
    )


@typer_app.command("plot")
def plot(
    results_folder: str = typer.Option(..., "--results_folder"),
    submission_folder: str = typer.Option(..., "--submission_folder"),
):
    PlotTask.run(results_folder, submission_folder)


if __name__ == "__main__":
    typer_app()
