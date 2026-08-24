"""Search parsed corpora for target-anchored dependency relation paths."""

import os
import re
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from SynFlow.const import DEFAULT_PATTERN, SENT_ID_PATTERN, VALID_FILLER_FORMATS
from SynFlow.utils import build_graph, format_filler


@dataclass(frozen=True)
class RelStep:
    """
    One relation step in a target-anchored dependency path.

    Attributes:
        relation: Directed dependency relation to traverse. This must include
            the ``chi_`` or ``pa_`` prefix produced by ``build_graph``.
        allowed_pos: POS tags allowed for the reached node. ``None`` means the
            step has no POS restriction, represented in the input as ``[]``.
    """

    relation: str
    allowed_pos: Optional[Set[str]] = None


RelPath = List[RelStep]
PathMatch = Tuple[List[Tuple[str, str, str, str]], str]


@dataclass(frozen=True)
class RelTemplate:
    """
    Parsed relation-search template.

    Attributes:
        paths: Required dependency paths from the target node. A target matches
            only when every path can be found from that same target.
    """

    paths: List[RelPath]


def build_context_lookup(
    sent_tokens: List[str],
    pattern: re.Pattern,
) -> Dict[str, Tuple[str, str, str]]:
    """
    Build a token-id to token, lemma, and POS lookup for one sentence.

    Args:
        sent_tokens: Parsed token lines for one sentence. Each line must match
            ``pattern`` and contain token, lemma, POS, ID, HEAD, DEPREL, and
            FEATS fields.
        pattern: Regular expression used to parse corpus token lines.

    Returns:
        Dictionary keyed by token ID. Values are ``(token, lemma, pos)`` tuples.
    """
    id2context = {}
    for line in sent_tokens:
        match = pattern.match(line)
        if not match:
            continue
        token, lemma, pos, idx, _, _, _ = match.groups()
        id2context[idx] = (token, lemma, pos)
    return id2context


def _strip_group(text: str) -> str:
    """
    Remove one surrounding pair of parentheses from a template component.

    Args:
        text: Raw template component, such as ``"(chi_obj, [NOUN])"``.

    Returns:
        The component without surrounding parentheses when they are present.
    """
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


def _parse_step(step_text: str) -> RelStep:
    """
    Parse one fixed-format relation step.

    Args:
        step_text: Step text in the required format
            ``"(relation, [POS1, POS2])"``. Use ``"(relation, [])"`` when the
            step has no POS restriction.

    Returns:
        A ``RelStep`` containing the relation and optional POS restriction.

    Raises:
        ValueError: If the step does not include both a relation and a POS list.
    """
    raw_step_text = step_text.strip()
    step_text = _strip_group(raw_step_text)
    if "," not in step_text:
        raise ValueError(
            "Relation steps must use the fixed format '(relation, [POS...])'. "
            f"Use an empty list for no POS restriction, e.g. '(chi_obj, [])'. Got: {raw_step_text!r}"
        )

    relation, pos_text = step_text.split(",", 1)
    relation = relation.strip()
    pos_text = pos_text.strip()
    if not relation or not pos_text.startswith("[") or not pos_text.endswith("]"):
        raise ValueError(
            "Relation steps must use the fixed format '(relation, [POS...])'. "
            f"Got: {raw_step_text!r}"
        )
    allowed_pos = {
        pos.strip()
        for pos in pos_text[1:-1].split(",")
        if pos.strip()
    }
    return RelStep(relation=relation, allowed_pos=allowed_pos or None)


def parse_rel_template(rel_template: str) -> RelTemplate:
    """
    Parse a relation-search template into required paths.

    Args:
        rel_template: Template string. ``&`` separates independent required
            paths, and ``>`` separates sequential steps inside one path. Every
            step must use ``"(relation, [POS...])"``. Use ``[]`` when there is
            no POS restriction.

            Examples:
            ``"> (chi_nsubj, [])"``
            ``"> (chi_nsubj, [NOUN, PROPN]) & > (chi_obj, [])"``

    Returns:
        Parsed relation template.

    Raises:
        ValueError: If the template is empty or contains a malformed step.
    """
    path_texts = [path.strip() for path in rel_template.split("&") if path.strip()]
    if not path_texts:
        raise ValueError("rel_template must include at least one relation path.")

    paths: List[RelPath] = []
    for path_text in path_texts:
        path_text = path_text.strip()
        if path_text.startswith(">"):
            path_text = path_text[1:].strip()
        steps = [_parse_step(step) for step in path_text.split(">") if step.strip()]
        if not steps:
            raise ValueError(f"Invalid empty relation path: {path_text!r}")
        paths.append(steps)

    return RelTemplate(paths=paths)


def _actual_next_relations(
    graph: Dict[str, List[str]],
    id2deprel: Dict[Tuple[str, str], str],
    node: str,
    parent_node: Optional[str],
) -> Set[str]:
    """
    Collect outgoing relation labels from a node, excluding the parent edge.

    Args:
        graph: Bidirectional dependency graph from ``build_graph``.
        id2deprel: Mapping from directed edge to dependency relation label.
        node: Token ID whose outward relations should be collected.
        parent_node: Previous token ID on the matched path. This reverse edge
            is ignored because it points back into the matched construction.

    Returns:
        Set of relation labels available from ``node`` away from ``parent_node``.
    """
    return {
        id2deprel[(node, nb)]
        for nb in graph.get(node, [])
        if nb != parent_node and id2deprel.get((node, nb))
    }


def _matches_exact_rel_tree(
    graph: Dict[str, List[str]],
    id2deprel: Dict[Tuple[str, str], str],
    target_id: str,
    path_match_group: Tuple[PathMatch, ...],
) -> bool:
    """
    Check whether selected path matches exhaust the local relation tree.

    ``close`` mode uses this function to restrict both width and depth. Every
    matched node may only have the outgoing relation labels explicitly present
    in the selected template paths. Leaf nodes in the template must be leaves
    relative to the outward search.

    Args:
        graph: Bidirectional dependency graph from ``build_graph``.
        id2deprel: Mapping from directed edge to dependency relation label.
        target_id: Token ID where relation search starts.
        path_match_group: One selected match for each required path.

    Returns:
        ``True`` when the selected matches exactly cover the outward tree under
        ``target_id``; otherwise ``False``.
    """
    allowed_relations_by_node: Dict[str, Set[str]] = {target_id: set()}
    parent_by_node: Dict[str, str] = {}

    for path_nodes, actual_path in path_match_group:
        previous_node = target_id
        path_relations = actual_path.split(" > ") if actual_path else []
        for relation, (node_id, _, _, _) in zip(path_relations, path_nodes):
            allowed_relations_by_node.setdefault(previous_node, set()).add(relation)
            allowed_relations_by_node.setdefault(node_id, set())
            parent_by_node[node_id] = previous_node
            previous_node = node_id

    for node, allowed_relations in allowed_relations_by_node.items():
        actual_relations = _actual_next_relations(
            graph,
            id2deprel,
            node,
            parent_by_node.get(node),
        )
        if actual_relations != allowed_relations:
            return False

    return True


def _dedupe_path_matches(path_matches: List[PathMatch]) -> List[PathMatch]:
    """
    Remove duplicate path matches while preserving order.

    Args:
        path_matches: Path matches selected for output.

    Returns:
        Path matches with duplicate node/path realizations removed.
    """
    seen: Set[Tuple[Tuple[Tuple[str, str], ...], str]] = set()
    deduped: List[PathMatch] = []
    for path_nodes, actual_path in path_matches:
        key = (tuple((node_id, path_to_node) for node_id, _, _, path_to_node in path_nodes), actual_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((path_nodes, actual_path))
    return deduped


def find_by_path(
    graph: Dict[str, List[str]],
    id2context: Dict[str, Tuple[str, str, str]],
    id2deprel: Dict[Tuple[str, str], str],
    target_id: str,
    path: RelPath,
    filler_format: str,
) -> List[PathMatch]:
    """
    Find matching fillers for one parsed path from one target node.

    Args:
        graph: Bidirectional dependency graph from ``build_graph``.
        id2context: Mapping from token ID to ``(token, lemma, pos)``.
        id2deprel: Mapping from directed edge to dependency relation label.
        target_id: Token ID where this path search starts.
        path: Parsed relation path to match.
        filler_format: Requested output format for reached nodes.

    Returns:
        Matching path realizations. Each result is
        ``([(node_id, filler, pos, path_to_node), ...], actual_path)``.
    """
    results: List[PathMatch] = []

    def dfs(
        node: str,
        depth: int,
        seen: Set[str],
        path_rels: List[str],
        path_nodes: List[Tuple[str, str, str, str]],
    ) -> None:
        """
        Recursively match the remaining relation steps.

        Args:
            node: Current token ID in the traversal.
            depth: Number of path steps already matched.
            seen: Token IDs already visited in this path.
            path_rels: Relation labels already matched.
            path_nodes: Matched nodes already reached, together with their
                formatted filler, POS tag, and relation path from the target.
        """
        if depth == len(path):
            results.append((path_nodes, " > ".join(path_rels)))
            return

        step = path[depth]
        for nb in graph.get(node, []):
            if nb in seen:
                continue
            lbl = id2deprel.get((node, nb))
            if lbl != step.relation:
                continue
            token, lemma, pos = id2context[nb]
            if step.allowed_pos is not None and pos not in step.allowed_pos:
                continue
            filler = format_filler(token, lemma, pos, lbl, filler_format)
            next_path_rels = path_rels + [lbl]
            dfs(
                nb,
                depth + 1,
                seen | {nb},
                next_path_rels,
                path_nodes + [(nb, filler, pos, " > ".join(next_path_rels))],
            )

    dfs(target_id, 0, {target_id}, [], [])
    return results


def process_file(
    args: Tuple[str, str, Optional[re.Pattern], str, str, str, str, str],
) -> List[Dict[str, object]]:
    """
    Process one corpus file for target-anchored relation matches.

    Args:
        args: Tuple containing ``(corpus_folder, fname, pattern, target_lemma,
            target_pos, rel_template, search_mode, filler_format)``.
            ``corpus_folder`` is the period/subfolder path containing ``fname``.
            ``pattern`` may be ``None`` to use ``DEFAULT_PATTERN``.

    Returns:
        List of output rows. Each row has ``sentence_id``, ``sentence``,
        ``sfillers``, and ``path`` keys.
    """
    corpus_folder, fname, pattern, target_lemma, target_pos, rel_template, search_mode, filler_format = args
    pattern = pattern or DEFAULT_PATTERN
    parsed_template = parse_rel_template(rel_template)
    has_multiple_paths = len(parsed_template.paths) > 1
    effective_search_mode = search_mode if has_multiple_paths else "open"
    results: List[Dict[str, object]] = []
    has_target_check_string = f"\t{target_lemma}\t{target_pos}"

    filepath = os.path.join(corpus_folder, fname)
    with open(filepath, encoding="utf8") as fh:
        sent_tokens: List[str] = []
        sent_forms: List[str] = []
        sent_id: Optional[str] = None
        has_target = False

        for line in fh:
            line = line.rstrip("\n")

            if line.startswith("<s id"):
                sent_tokens = []
                sent_forms = []
                has_target = False
                match = SENT_ID_PATTERN.match(line)
                sent_id = match.group(1) if match else None

            elif line.startswith("</s>"):
                if sent_tokens and has_target:
                    id2wp, graph, id2deprel = build_graph(sent_tokens, pattern)
                    id2context = build_context_lookup(sent_tokens, pattern)
                    sentence_text = " ".join(sent_forms)
                    target_lp = f"{target_lemma}/{target_pos}"
                    target_ids = [idx for idx, lemma_pos in id2wp.items() if lemma_pos == target_lp]

                    for target_id in target_ids:
                        path_matches_by_path: List[List[PathMatch]] = []
                        found_all = True
                        for path in parsed_template.paths:
                            matches = find_by_path(
                                graph,
                                id2context,
                                id2deprel,
                                target_id,
                                path,
                                filler_format,
                            )
                            if not matches:
                                found_all = False
                                break
                            path_matches_by_path.append(matches)

                        if not found_all:
                            continue

                        if effective_search_mode == "close":
                            valid_match_groups = [
                                match_group
                                for match_group in product(*path_matches_by_path)
                                if _matches_exact_rel_tree(graph, id2deprel, target_id, match_group)
                            ]
                            if not valid_match_groups:
                                continue
                            path_matches = _dedupe_path_matches([
                                path_match
                                for match_group in valid_match_groups
                                for path_match in match_group
                            ])
                        elif effective_search_mode == "closeh":
                            required_direct = {path[0].relation for path in parsed_template.paths}
                            actual_direct = {
                                id2deprel.get((target_id, nb))
                                for nb in graph.get(target_id, [])
                                if id2deprel.get((target_id, nb))
                            }
                            if actual_direct != required_direct:
                                continue
                            path_matches = _dedupe_path_matches([
                                path_match
                                for matches in path_matches_by_path
                                for path_match in matches
                            ])
                        else:
                            path_matches = _dedupe_path_matches([
                                path_match
                                for matches in path_matches_by_path
                                for path_match in matches
                            ])

                        for path_nodes, _ in path_matches:
                            for _, filler, _, path_to_node in path_nodes:
                                results.append(
                                    {
                                        "sentence_id": sent_id,
                                        "sentence": sentence_text,
                                        "sfillers": [filler],
                                        "path": path_to_node,
                                    }
                                )

                sent_tokens = []
                sent_forms = []

            else:
                sent_tokens.append(line)
                match = pattern.match(line)
                if match:
                    sent_forms.append(match.group(1))
                if has_target_check_string in line:
                    has_target = True

    return results


def rel_explorer(
    corpus_folder: str,
    pattern: re.Pattern = None,
    target_lemma: str = None,
    target_pos: str = None,
    deprel: str = None,
    search_mode: str = "open",
    filler_format: str = "lemma/pos",
    num_processes: int = max(1, cpu_count() - 1),
) -> pd.DataFrame:
    """
    Search a corpus for relation paths starting from a target lemma/POS.

    Args:
        corpus_folder: Root corpus folder. The function expects period or
            subcorpus directories inside this root, and each subdirectory may
            contain ``.txt`` or ``.conllu`` parsed files.
        pattern: Regex used to parse token lines. Defaults to
            ``DEFAULT_PATTERN``.
        target_lemma: Lemma of the target node where each search starts.
        target_pos: POS tag of the target node where each search starts.
        deprel: Relation template such as
            ``"> (chi_nsubj, [NOUN, PROPN]) & > (chi_obj, [])"``. ``&``
            separates independent required paths, and ``>`` separates
            sequential steps inside one path. Every step must include a POS
            restriction list; use ``[]`` for no restriction.
        search_mode: ``"open"``, ``"close"``, or ``"closeh"``. When ``deprel``
            contains only one path, the effective mode is always ``"open"``.
            With multiple paths, ``open`` accepts targets that include at least
            the required paths. ``close`` restricts both width and depth: every
            matched node may only have outgoing relations explicitly listed in
            the template. ``closeh`` restricts horizontal width only at the
            target: the target's direct relation set must match exactly, while
            deeper specialization under those direct slots is allowed.
        filler_format: Format for context fillers. Must be one of
            ``VALID_FILLER_FORMATS``.
        num_processes: Number of worker processes. Use ``1`` for inline,
            deterministic execution during debugging.

    Returns:
        DataFrame with columns ``sentence_id``, ``sentence``, ``sfillers``, and
        ``path``. One row is returned for each matched filler node.

    Raises:
        ValueError: If required arguments are missing, ``search_mode`` is not
        supported, ``filler_format`` is invalid, or the relation template is
        malformed.
    """
    if not os.path.isdir(corpus_folder):
        raise ValueError(f"Corpus folder does not exist: {corpus_folder}")
    if target_lemma is None or target_pos is None or deprel is None:
        raise ValueError("'target_lemma', 'target_pos', and 'deprel' must be provided.")
    if search_mode not in {"open", "close", "closeh"}:
        raise ValueError("search_mode must be one of: open, close, closeh")
    if filler_format not in VALID_FILLER_FORMATS:
        valid_formats = ", ".join(sorted(VALID_FILLER_FORMATS))
        raise ValueError(f"filler_format must be one of: {valid_formats}")

    pattern = pattern or DEFAULT_PATTERN
    num_procs = max(1, num_processes)
    rows: List[Dict[str, object]] = []

    for subfolder in sorted(os.listdir(corpus_folder)):
        subfolder_path = os.path.join(corpus_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        files = [
            fname
            for fname in os.listdir(subfolder_path)
            if fname.endswith((".conllu", ".txt"))
        ]
        args = [
            (subfolder_path, fname, pattern, target_lemma, target_pos, deprel, search_mode, filler_format)
            for fname in files
        ]
        if num_procs == 1:
            file_results = (process_file(arg) for arg in args)
        else:
            pool = Pool(num_procs)
            file_results = pool.imap_unordered(process_file, args, chunksize=10)
        try:
            for file_rows in file_results:
                rows.extend(file_rows)
        finally:
            if num_procs != 1:
                pool.close()
                pool.join()

    return pd.DataFrame(rows, columns=["sentence_id", "sentence", "sfillers", "path"])
