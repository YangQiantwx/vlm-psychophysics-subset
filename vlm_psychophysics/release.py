"""Immutable dataset identity and validated manifest paths."""

from pathlib import PurePosixPath


REPO_ID = "DrQY/vlm-psychophysics-subset"


REVISION = "f562d61b4703b744c2f00f66a7ebabeacc4e0a0a"


SCOPES = tuple(
    f"{domain}/t{task}" for domain in ("classic", "natural") for task in range(1, 6)
)


PAIRED_SCOPES = ("classic/t5/paired-v1", "natural/t5/paired-v1")


def payload_path(scope, name):
    path = PurePosixPath(name)
    if (
        "\\" in name
        or path.as_posix() != name
        or path.is_absolute()
        or ".." in path.parts
        or not path.is_relative_to(PurePosixPath(scope))
        or path == PurePosixPath(scope)
        or any(char in name for char in "*?[]")
    ):
        raise ValueError(f"Unexpected {scope} release path: {name}")
    return name


def select_samples(trials, selectors):
    """Resolve exact trial IDs, video filenames or dataset-relative video paths."""
    wanted = dict.fromkeys(selectors)
    matches = {value: set() for value in wanted}
    for scope, sample_id, video_path in trials:
        keys = {sample_id, video_path, PurePosixPath(video_path).name}
        for value in keys & wanted.keys():
            matches[value].add((scope, sample_id))
    selected = set()
    for value, candidates in matches.items():
        if not candidates:
            raise ValueError(f"No published trial matches {value!r} in the selected tasks.")
        if len(candidates) > 1:
            tasks = set()
            for scope, _ in candidates:
                domain, task, *_ = scope.split("/")
                tasks.add(f"{task}-{domain}")
            task_names = ", ".join(sorted(tasks))
            raise ValueError(
                f"{value!r} matches multiple published trials in {task_names}. "
                "Use --task to choose one task or provide an exact sample ID or dataset path."
            )
        selected.update(candidates)
    return selected
