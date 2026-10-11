"""Local mechanism catalog; never downloads or substitutes scientific inputs."""
import argparse
import json
import os
from pathlib import Path


CATALOG = (
    ('H2-O2', 'h2o2.yaml', 10, 29, 'bundled', ('h2o2',)),
    ('DRM19', 'drm19.yaml', 21, 84, 'bundled', ('drm19',)),
    ('DRM22', 'drm22.yaml', 24, 104, 'bundled', ('drm22',)),
    ('FFCM-1', 'ffcm1.yaml', 38, 291, 'bundled', ('ffcm1',)),
    ('GRI-Mech 3.0', 'gri30.yaml', 53, 325, 'bundled', ('grimech30', 'gri30')),
    ('USC Mech II', 'uscmech2.yaml', 111, 784, 'bundled', ('uscmechii', 'uscmech2')),
    ('JetSurF 2.0', 'jetsurf2.yaml', 348, 2163, 'bundled', ('jetsurf20', 'jetsurf2')),
    ('AramcoMech 2.0', 'aramcomech2.yaml', 493, 2716, 'local', ('aramcomech20', 'aramcomech2')),
    ('AramcoMech 3.0', 'aramcomech3.yaml', 581, 3037, 'local', ('aramcomech30', 'aramcomech3')),
)


def local_mechanism_directory():
    configured = os.environ.get('KFLAME_MECHANISM_DIR')
    return Path(configured).expanduser() if configured else Path.home()/'.kflame'/'mechanisms'


def _normalized(name):
    return ''.join(character for character in name.lower() if character.isalnum())


def catalog_path(name):
    """Known names resolve to their own file only; explicit paths use the reader."""
    key = _normalized(str(name).removesuffix('.yaml'))
    for title, filename, _, _, location, aliases in CATALOG:
        if key in (*aliases, _normalized(title), _normalized(Path(filename).stem)):
            directory = (Path(__file__).parent/'data' if location == 'bundled'
                         else local_mechanism_directory())
            path = directory/filename
            if not path.is_file():
                raise FileNotFoundError(f'{title} is not installed: {path}. '
                    'Supply an explicit YAML path or set KFLAME_MECHANISM_DIR to its directory.')
            return path.resolve()
    return None


def available_mechanisms():
    result = []
    for title, filename, species, reactions, location, _ in CATALOG:
        directory = Path(__file__).parent/'data' if location == 'bundled' else local_mechanism_directory()
        path = directory/filename
        result.append(dict(name=title, filename=filename, species=species, reactions=reactions,
                           location=location, installed=path.is_file(), path=str(path)))
    return result


def main():
    parser = argparse.ArgumentParser(description='List installed native chemistry inputs')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    rows = available_mechanisms()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f'{"Mechanism":<18} {"Species":>7} {"Reactions":>9}  Status')
        for row in rows:
            status = row['location'] if row['installed'] else 'not installed'
            print(f'{row["name"]:<18} {row["species"]:>7} {row["reactions"]:>9}  {status}')
