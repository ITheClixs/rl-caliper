"""Run manifests name the code and configuration that produced a result, portably."""

from caliper.runtime import io


def test_paths_inside_the_repository_are_recorded_relative_to_it():
    inside = str(io.ROOT / "runs" / "s11_prospective")
    assert io.portable({"store": inside}) == {"store": "runs/s11_prospective"}


def test_paths_outside_the_repository_keep_only_their_last_component():
    config = {"store": "/private/tmp/someone/scratchpad/smoke", "nested": ["/opt/models/x"]}
    assert io.portable(config) == {
        "store": "<outside the repository>/smoke",
        "nested": ["<outside the repository>/x"],
    }


def test_relative_values_and_numbers_are_untouched_so_existing_hashes_do_not_move():
    config = {"store": "runs/p2", "pool": 24, "steps": [10, 25], "label": "a/b"}
    assert io.portable(config) == config
    assert io.config_hash(io.portable(config)) == io.config_hash(config)


def test_a_manifest_says_whether_the_tree_was_dirty():
    record = io.manifest({"store": "runs/p2"})
    assert record["git_dirty"] in (True, False, None)
    assert record["config_hash"] == io.config_hash({"store": "runs/p2"})
