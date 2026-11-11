"""Mutation for 3jza_superset_plant: remove one registry json so superset fails."""
import os

MUTATIONS = {"3jza_superset_plant": "3jza superset guard plant: drop one registry shard"}


def apply(root):
    path = os.path.join(root, "scripts", "fs2y_mutations.d", "3v6d2_link_calendar.json")
    backup = open(path).read()
    os.remove(path)
    return backup, path
