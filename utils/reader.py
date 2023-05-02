from pybrat.parser import BratParser, Example, Reference
from pybrat.utils import iter_file_groups
from typing import Optional, Iterable, Union
import collections
import os
import re

SPACES_PATTERN = re.compile(r'\xa0')


class BratParserPlus(BratParser):
    def __init__(self, ignore_types: Optional[Iterable[str]] = None, error: str = "raise"):
        super().__init__(ignore_types, error)

    def _parse_ann(self, ann, encoding, spaces_pattern):
        # Parser entities and store required data for parsing relations
        # and events.
        entity_matches, relation_matches, event_matches = [], [], []
        references = collections.defaultdict(list)

        with open(ann, mode="r", encoding=encoding) as f:
            for line in f:
                if spaces_pattern is not None:
                    line = re.sub(spaces_pattern, ' ', line)

                line = line.rstrip()
                if not line or line.startswith("#") or self._should_ignore_line(line):
                    continue

                if line.startswith("T"):
                    if match := self._parse_entity(line):
                        entity_matches += [match]
                elif line.startswith("R"):
                    if match := self._parse_relation(line):
                        relation_matches += [match]
                elif line.startswith("*"):
                    if match := self._parse_equivalence_relations(line):
                        relation_matches += list(match)
                elif line.startswith("E"):
                    if match := self._parse_event(line):
                        event_matches += [match]
                elif line.startswith("N"):
                    if match := self._parse_reference(line):
                        references[match["entity"]] += [
                            Reference(
                                rid=match["rid"],
                                eid=match["eid"],
                                entry=match["entry"],
                                id=match["id"],
                            )
                        ]
                elif line.startswith("AM"):
                    raise NotImplementedError()

        # Format entities.
        entities = self._format_entities(entity_matches, references)
        self._check_entities(entities.values())

        # Format relations.
        relations = self._format_relations(relation_matches, entities)

        # Format events.
        events = self._format_events(event_matches, entities)

        return {
            "entities": list(entities.values()),
            "relations": relations,
            "events": list(events.values()),
        }

    def _parse_text(self, txt, encoding, spaces_pattern):  # pylint: disable=no-self-use
        with open(txt, mode="r", encoding=encoding) as f:
            text = f.read()
            if spaces_pattern is not None:
                text = re.sub(spaces_pattern, ' ', text)
            return text

    def parse(
            self, dirname: Union[str, bytes, os.PathLike], encoding: str = "utf-8",
            spaces_pattern: Optional[re.Pattern] = None
    ) -> list[Example]:
        """Parse examples in given directory.

        Args:
            dirname (Union[str, bytes, os.PathLike]): Directory
                containing brat examples.
            encoding (str): Encoding for reading text files and
                ann files
            spaces_pattern (Optional[re.Pattern]): Pattern of spaces to be replaced with a single space

        Returns:
            examples (list[Example]): Parsed examples.
        """

        examples = []

        file_groups = iter_file_groups(
            dirname,
            self.exts,
            missing="error" if self.error == "raise" else "ignore",
        )

        for key, (ann_file, txt_file) in file_groups:
            txt = self._parse_text(txt_file, encoding=encoding, spaces_pattern=spaces_pattern)
            ann = self._parse_ann(ann_file, encoding=encoding, spaces_pattern=spaces_pattern)
            examples += [Example(text=txt, **ann, id=key)]

        examples.sort(key=lambda x: x.id if x.id is not None else "")

        return examples
